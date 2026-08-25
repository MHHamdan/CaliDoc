"""
Donut Encoder for OCR-free Document Understanding

Donut (Document Understanding Transformer) is an end-to-end transformer
that directly maps document images to outputs without explicit OCR.

Key advantages for DocVQA:
- No OCR dependency (avoids OCR errors)
- Generative approach (can handle non-extractive answers)
- Pre-trained on document understanding tasks

Reference: https://arxiv.org/abs/2111.15664
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from PIL import Image
import re

from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
)


@dataclass
class DonutOutput:
    """Output from Donut model."""
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    generated_text: Optional[List[str]] = None
    hidden_states: Optional[torch.Tensor] = None
    encoder_hidden_states: Optional[torch.Tensor] = None
    cross_attentions: Optional[torch.Tensor] = None


class DonutForDocVQA(nn.Module):
    """
    Donut model fine-tuned for Document Visual Question Answering.

    Uses generative approach: given image + question, generates answer text.

    Args:
        model_name: Pretrained Donut model name
        max_length: Maximum generation length
        image_size: Input image size
    """

    def __init__(
        self,
        model_name: str = "naver-clova-ix/donut-base-finetuned-docvqa",
        max_length: int = 128,
        image_size: Tuple[int, int] = (1280, 960),
    ):
        super().__init__()

        self.max_length = max_length
        self.image_size = image_size

        # Load pretrained model and processor
        self.processor = DonutProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

        # Set decoder start token for VQA task
        self.model.config.decoder_start_token_id = self.processor.tokenizer.convert_tokens_to_ids(['<s_docvqa>'])[0]

        # Ensure pad token is set
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            self.model.config.pad_token_id = self.processor.tokenizer.eos_token_id

    def forward(
        self,
        pixel_values: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> DonutOutput:
        """
        Forward pass for training.

        Args:
            pixel_values: Processed images [batch, 3, H, W]
            decoder_input_ids: Input IDs for decoder
            labels: Target token IDs for loss computation
            attention_mask: Encoder attention mask
            decoder_attention_mask: Decoder attention mask
        """
        outputs = self.model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )

        return DonutOutput(
            logits=outputs.logits,
            loss=outputs.loss,
            hidden_states=outputs.decoder_hidden_states[-1] if outputs.decoder_hidden_states else None,
            encoder_hidden_states=outputs.encoder_last_hidden_state,
            cross_attentions=outputs.cross_attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        questions: Optional[List[str]] = None,
        max_new_tokens: int = 128,
        num_beams: int = 1,
        **kwargs
    ) -> List[str]:
        """
        Generate answers for document images.

        Args:
            pixel_values: Processed images
            questions: List of questions (used for prompt formatting)
            max_new_tokens: Maximum tokens to generate
            num_beams: Beam search width

        Returns:
            List of generated answer strings
        """
        # Create decoder prompt with question
        if questions is not None:
            prompts = [f"<s_docvqa><s_question>{q}</s_question><s_answer>" for q in questions]
            prompt_ids = self.processor.tokenizer(
                prompts,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
            ).input_ids.to(pixel_values.device)
        else:
            # Default prompt
            prompt_ids = torch.tensor(
                [[self.model.config.decoder_start_token_id]],
                device=pixel_values.device
            ).expand(pixel_values.size(0), -1)

        # Generate
        generated_ids = self.model.generate(
            pixel_values,
            decoder_input_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **kwargs
        )

        # Decode and post-process
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=False)
        answers = [self._extract_answer(text) for text in generated_texts]

        return answers

    def _extract_answer(self, text: str) -> str:
        """Extract answer from generated text."""
        # Pattern: <s_answer>ANSWER</s_answer> or just content after <s_answer>
        match = re.search(r'<s_answer>(.*?)(?:</s_answer>|</s>|$)', text, re.DOTALL)
        if match:
            answer = match.group(1).strip()
        else:
            # Fallback: take text after last special token
            answer = re.sub(r'<[^>]+>', '', text).strip()
        return answer

    def prepare_inputs(
        self,
        images: List[Image.Image],
        questions: List[str],
        answers: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for training or inference.

        Args:
            images: List of PIL images
            questions: List of questions
            answers: List of answers (for training)

        Returns:
            Dictionary with model inputs
        """
        # Process images
        pixel_values = self.processor(
            images,
            return_tensors="pt",
        ).pixel_values

        if answers is not None:
            # Training: create decoder inputs and labels
            prompts = [
                f"<s_docvqa><s_question>{q}</s_question><s_answer>{a}</s_answer></s>"
                for q, a in zip(questions, answers)
            ]

            encoding = self.processor.tokenizer(
                prompts,
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            # Shift for labels (predict next token)
            labels = encoding.input_ids.clone()
            # Mask prompt tokens (don't compute loss on question)
            for i, (q, a) in enumerate(zip(questions, answers)):
                prompt_len = len(self.processor.tokenizer.encode(
                    f"<s_docvqa><s_question>{q}</s_question><s_answer>",
                    add_special_tokens=False
                ))
                labels[i, :prompt_len] = -100

            return {
                'pixel_values': pixel_values,
                'decoder_input_ids': encoding.input_ids,
                'decoder_attention_mask': encoding.attention_mask,
                'labels': labels,
            }
        else:
            # Inference: just return images and question prompts
            return {
                'pixel_values': pixel_values,
                'questions': questions,
            }


class DonutDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for Donut training on DocVQA.
    """

    def __init__(
        self,
        dataset,
        processor: DonutProcessor,
        max_length: int = 128,
        split: str = 'train',
    ):
        self.dataset = dataset
        self.processor = processor
        self.max_length = max_length
        self.split = split

        # Task prompt
        self.task_start = "<s_docvqa>"
        self.question_start = "<s_question>"
        self.question_end = "</s_question>"
        self.answer_start = "<s_answer>"
        self.answer_end = "</s_answer>"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]

        # Get image
        if isinstance(item['image'], str):
            image = Image.open(item['image']).convert('RGB')
        else:
            image = item['image'].convert('RGB')

        # Process image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)

        # Get question and answer
        question = item['question']
        answers = item.get('answers', [])
        answer = answers[0] if answers else ''

        # Create target sequence
        target_sequence = (
            f"{self.task_start}"
            f"{self.question_start}{question}{self.question_end}"
            f"{self.answer_start}{answer}{self.answer_end}"
            f"{self.processor.tokenizer.eos_token}"
        )

        # Tokenize
        encoding = self.processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )

        labels = encoding.input_ids.squeeze(0).clone()

        # Mask question part for loss (only predict answer)
        question_prompt = (
            f"{self.task_start}"
            f"{self.question_start}{question}{self.question_end}"
            f"{self.answer_start}"
        )
        prompt_len = len(self.processor.tokenizer.encode(
            question_prompt, add_special_tokens=False
        ))
        labels[:prompt_len] = -100

        # Also mask padding
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            'pixel_values': pixel_values,
            'decoder_input_ids': encoding.input_ids.squeeze(0),
            'decoder_attention_mask': encoding.attention_mask.squeeze(0),
            'labels': labels,
            'question': question,
            'answer': answer,
            'question_id': item.get('question_id', item.get('questionId', idx)),
        }


def create_donut_docvqa_dataloader(
    dataset,
    processor: DonutProcessor,
    batch_size: int = 4,
    num_workers: int = 4,
    split: str = 'train',
    max_length: int = 128,
) -> torch.utils.data.DataLoader:
    """Create DataLoader for Donut DocVQA training."""

    donut_dataset = DonutDataset(
        dataset=dataset,
        processor=processor,
        max_length=max_length,
        split=split,
    )

    def collate_fn(batch):
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        decoder_input_ids = torch.stack([item['decoder_input_ids'] for item in batch])
        decoder_attention_mask = torch.stack([item['decoder_attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])

        return {
            'pixel_values': pixel_values,
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': decoder_attention_mask,
            'labels': labels,
            'questions': [item['question'] for item in batch],
            'answers': [item['answer'] for item in batch],
            'question_ids': [item['question_id'] for item in batch],
        }

    return torch.utils.data.DataLoader(
        donut_dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
