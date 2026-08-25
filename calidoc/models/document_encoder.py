"""
Document Encoder Wrappers for CaliDoc

This module provides unified interfaces for various document understanding models:
- LayoutLMv3 (Microsoft): OCR-dependent, strong on structured documents
- DocOwl 1.5 (Alibaba): OCR-free VLM, 7B parameters
- GOT-OCR 2.0 (StepFun): OCR VLM, specialized for OCR tasks

Each encoder provides:
- Standardized input/output interface
- Access to intermediate representations for uncertainty estimation
- Layout attention weights for structure-aware conformal prediction
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from transformers import (
    LayoutLMv3ForQuestionAnswering,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
)
from PIL import Image
import numpy as np


@dataclass
class EncoderOutput:
    """Standardized output from document encoders."""
    logits: torch.Tensor                          # Model predictions
    hidden_states: Optional[torch.Tensor] = None  # Last hidden states
    attentions: Optional[torch.Tensor] = None     # Attention weights
    layout_attention: Optional[torch.Tensor] = None  # Layout-specific attention
    ocr_confidence: Optional[torch.Tensor] = None    # OCR confidence scores
    loss: Optional[torch.Tensor] = None           # Loss from model (if labels provided)
    start_logits: Optional[torch.Tensor] = None   # Start logits for VQA
    end_logits: Optional[torch.Tensor] = None     # End logits for VQA


class DocumentEncoder(ABC, nn.Module):
    """Abstract base class for document encoders."""

    @abstractmethod
    def forward(
        self,
        images: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        bbox: Optional[torch.Tensor] = None,
        **kwargs
    ) -> EncoderOutput:
        """Forward pass through the encoder."""
        pass

    @abstractmethod
    def get_layout_attention(self) -> Optional[torch.Tensor]:
        """Extract layout-specific attention weights."""
        pass


class LayoutLMv3Encoder(DocumentEncoder):
    """
    LayoutLMv3 wrapper for document understanding.

    LayoutLMv3 is a multi-modal transformer that takes:
    - Document images
    - OCR text
    - Bounding boxes

    It's particularly strong on structured documents like forms and receipts.
    """

    def __init__(
        self,
        model_name: str = "microsoft/layoutlmv3-large",
        task: str = "vqa",  # 'vqa' or 'ner'
        num_labels: Optional[int] = None,
        output_attentions: bool = True,
        output_hidden_states: bool = True,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()

        self.task = task
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states

        # Load processor
        self.processor = LayoutLMv3Processor.from_pretrained(
            model_name,
            apply_ocr=False  # We'll handle OCR separately
        )

        # Load model based on task
        if task == "vqa":
            self.model = LayoutLMv3ForQuestionAnswering.from_pretrained(
                model_name,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        elif task == "ner":
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        else:
            raise ValueError(f"Unknown task: {task}")

        # Enable gradient checkpointing for memory efficiency (only if supported)
        if gradient_checkpointing:
            try:
                self.model.gradient_checkpointing_enable()
            except ValueError:
                # Some model variants don't support gradient checkpointing
                pass

        # Store last attention for uncertainty estimation
        self._last_attention = None

    def forward(
        self,
        images: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        bbox: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> EncoderOutput:
        """
        Forward pass through LayoutLMv3.

        Args:
            images: Document images (if pixel_values not provided)
            input_ids: Tokenized text input
            attention_mask: Attention mask
            bbox: Bounding boxes for each token
            pixel_values: Processed image tensors
            start_positions: Start positions for VQA
            end_positions: End positions for VQA
            labels: Labels for NER
        """
        # Use pixel_values if provided, otherwise process images
        if pixel_values is None and images is not None:
            # Assume images are already preprocessed
            pixel_values = images

        # Forward through model
        loss = None
        start_logits = None
        end_logits = None

        if self.task == "vqa":
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
                start_positions=start_positions,
                end_positions=end_positions,
                output_attentions=self.output_attentions,
                output_hidden_states=self.output_hidden_states,
            )
            # Store separate start/end logits for evaluation
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            # Stack for unified interface
            logits = torch.stack([outputs.start_logits, outputs.end_logits], dim=-1)
            # Extract loss if computed (when start_positions/end_positions provided)
            loss = getattr(outputs, 'loss', None)
        else:  # NER
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
                labels=labels,
                output_attentions=self.output_attentions,
                output_hidden_states=self.output_hidden_states,
            )
            logits = outputs.logits
            # Extract loss if computed (when labels provided)
            loss = getattr(outputs, 'loss', None)

        # Extract attention weights
        attentions = None
        if self.output_attentions and hasattr(outputs, 'attentions'):
            attentions = outputs.attentions
            self._last_attention = attentions

        # Extract hidden states
        hidden_states = None
        if self.output_hidden_states and hasattr(outputs, 'hidden_states'):
            hidden_states = outputs.hidden_states[-1]  # Last layer

        # Compute layout attention (aggregate over visual tokens)
        layout_attention = self.get_layout_attention()

        return EncoderOutput(
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
            layout_attention=layout_attention,
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
        )

    def get_layout_attention(self) -> Optional[torch.Tensor]:
        """
        Extract layout-specific attention weights.

        Returns attention weights aggregated over visual patch tokens,
        which indicates how the model attends to different spatial regions.
        """
        if self._last_attention is None:
            return None

        # LayoutLMv3 has 49 visual tokens (7x7 patches)
        # Aggregate attention to these tokens
        last_layer_attention = self._last_attention[-1]  # [batch, heads, seq, seq]

        # Average over heads
        attention = last_layer_attention.mean(dim=1)  # [batch, seq, seq]

        # Sum attention to visual tokens (indices depend on model config)
        # Typically first 49 tokens are visual
        visual_attention = attention[:, :, :49].sum(dim=-1)  # [batch, seq]

        return visual_attention

    def prepare_inputs(
        self,
        images: List[Image.Image],
        texts: List[str],
        boxes: Optional[List[List[List[int]]]] = None,
        words: Optional[List[List[str]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the model using the processor.

        Args:
            images: List of PIL images
            texts: List of text strings (questions for VQA)
            boxes: Bounding boxes for each word
            words: OCR words

        Returns:
            Dictionary of model inputs
        """
        if words is not None and boxes is not None:
            encoding = self.processor(
                images=images,
                text=texts,
                boxes=boxes,
                word_labels=words,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
        else:
            encoding = self.processor(
                images=images,
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

        return encoding


class DocOwlEncoder(DocumentEncoder):
    """
    DocOwl 1.5 wrapper for OCR-free document understanding.

    DocOwl is a 7B parameter vision-language model that can understand
    documents without explicit OCR, using structure-aware visual encoding.
    """

    def __init__(
        self,
        model_name: str = "mPLUG/DocOwl1.5",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_flash_attention: bool = True,
        max_length: int = 2048,
    ):
        super().__init__()

        self.max_length = max_length

        # Quantization config
        quantization_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif load_in_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            quantization_config=quantization_config,
            attn_implementation="flash_attention_2" if use_flash_attention else None,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self._last_attention = None

    def forward(
        self,
        images: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> EncoderOutput:
        """Forward pass through DocOwl."""
        if pixel_values is None:
            pixel_values = images

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            output_attentions=True,
            output_hidden_states=True,
        )

        logits = outputs.logits

        # Extract attention and hidden states
        attentions = getattr(outputs, 'attentions', None)
        hidden_states = getattr(outputs, 'hidden_states', None)
        if hidden_states:
            hidden_states = hidden_states[-1]

        self._last_attention = attentions

        return EncoderOutput(
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
            layout_attention=self.get_layout_attention(),
        )

    def get_layout_attention(self) -> Optional[torch.Tensor]:
        """Extract layout attention from DocOwl."""
        if self._last_attention is None:
            return None
        # DocOwl uses visual tokens in the attention
        # Implementation depends on model specifics
        return None

    def generate(
        self,
        images: List[Image.Image],
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs
    ) -> List[str]:
        """
        Generate text responses for document images.

        Args:
            images: List of document images
            prompts: List of text prompts/questions
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            List of generated responses
        """
        responses = []
        for image, prompt in zip(images, prompts):
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    **kwargs
                )

            response = self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True
            )
            responses.append(response)

        return responses


class GOTOCREncoder(DocumentEncoder):
    """
    GOT-OCR 2.0 wrapper for OCR-focused document understanding.

    GOT-OCR is a 580M parameter model specialized for OCR tasks,
    supporting handwriting, formulas, tables, and more.
    """

    def __init__(
        self,
        model_name: str = "stepfun-ai/GOT-OCR2_0",
        use_flash_attention: bool = True,
        max_length: int = 8192,
    ):
        super().__init__()

        self.max_length = max_length

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if use_flash_attention else None,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self._last_attention = None

    def forward(
        self,
        images: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> EncoderOutput:
        """Forward pass through GOT-OCR."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            output_attentions=True,
            output_hidden_states=True,
            **kwargs
        )

        return EncoderOutput(
            logits=outputs.logits,
            hidden_states=outputs.hidden_states[-1] if outputs.hidden_states else None,
            attentions=outputs.attentions,
        )

    def get_layout_attention(self) -> Optional[torch.Tensor]:
        return None

    def ocr(
        self,
        images: List[Image.Image],
        ocr_type: str = 'ocr',  # 'ocr', 'format', 'fine-grained'
        ocr_box: str = '',
        ocr_color: str = '',
    ) -> List[str]:
        """
        Perform OCR on document images.

        Args:
            images: List of document images
            ocr_type: Type of OCR ('ocr', 'format', 'fine-grained')
            ocr_box: Box coordinates for region OCR
            ocr_color: Color for fine-grained OCR

        Returns:
            List of OCR text results
        """
        results = []
        for image in images:
            # GOT-OCR uses chat-based interface
            result = self.model.chat(
                self.tokenizer,
                image,
                ocr_type=ocr_type,
                ocr_box=ocr_box,
                ocr_color=ocr_color
            )
            results.append(result)

        return results


def create_encoder(
    encoder_type: str,
    **kwargs
) -> DocumentEncoder:
    """
    Factory function to create document encoders.

    Args:
        encoder_type: Type of encoder ('layoutlmv3', 'docowl', 'got_ocr')
        **kwargs: Additional arguments for the encoder

    Returns:
        Initialized document encoder
    """
    encoders = {
        'layoutlmv3': LayoutLMv3Encoder,
        'docowl': DocOwlEncoder,
        'got_ocr': GOTOCREncoder,
    }

    if encoder_type not in encoders:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    return encoders[encoder_type](**kwargs)
