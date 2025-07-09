import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
import torch
import io
import base64
from types import SimpleNamespace

# Add imports for the functions to be tested
from utils.prompt_utils import _gpt_huggingface, load_gpt_config_from_file, init_client

class TestGptHuggingface(unittest.TestCase):

    @patch('utils.prompt_utils.load_pil_images')
    @patch('torch.cuda.empty_cache')
    @patch('utils.prompt_utils.client', new_callable=MagicMock)
    def test_gpt_huggingface_flow(self, mock_client, mock_empty_cache, mock_load_pil):
        """
        Test the end-to-end flow of the _gpt_huggingface function,
        mocking the client and its components (processor, model).
        """
        # Arrange
        # 1. Create dummy inputs
        dummy_image = Image.new('RGB', (60, 30), color='red')
        dummy_prompt = "Describe the image."
        expected_description = "A description of a red image."
        mock_load_pil.return_value = [dummy_image]

        # 2. Configure the mock client and its components
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # The processor returns mock inputs for the model
        mock_prepare_inputs = SimpleNamespace(
            attention_mask=torch.tensor([[1, 1, 1]]),
            # Add other necessary attributes if the tested function requires them
        )
        mock_processor.return_value.to.return_value = mock_prepare_inputs

        # The model prepares inputs_embeds
        mock_model.prepare_inputs_embeds.return_value = "mock_embeds"

        # The model returns mock output tensors
        mock_output_tensors = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.generate.return_value = mock_output_tensors
        mock_model.device = 'cuda'

        # The tokenizer decodes the model output to the final string
        mock_tokenizer.decode.return_value = expected_description
        mock_processor.tokenizer = mock_tokenizer

        # Assign the mocks to the client dictionary
        mock_client.__getitem__.side_effect = lambda key: {'processor': mock_processor, 'model': mock_model}[key]

        # Act
        result = _gpt_huggingface(dummy_image, dummy_prompt)

        # Assert
        self.assertEqual(result, expected_description)
        mock_empty_cache.assert_called_once()
        mock_model.prepare_inputs_embeds.assert_called_once_with(**mock_prepare_inputs)
        mock_model.generate.assert_called_once()
        mock_tokenizer.decode.assert_called_with(mock_output_tensors[0].cpu().tolist(), skip_special_tokens=True)

    def test_gpt_huggingface_integration(self):
        """
        Tests the _gpt_huggingface function with the actual Hugging Face model.
        This is a slow integration test that requires network access and a GPU.
        """
        # Arrange
        # Set up the configuration and initialize the client for Hugging Face
        load_gpt_config_from_file()
        init_client() # This will initialize the global 'client' variable

        from utils.prompt_utils import client # import client after it's initialized
        model_dtype = client['model'].dtype

        # Create a simple, real image and cast it to the model's dtype
        dummy_image = Image.new('RGB', (100, 100), color = 'blue')
        dummy_prompt = "What is the main color of this image?"

        # Act
        # This will call the actual function, which downloads and runs the model.
        result = _gpt_huggingface(dummy_image, dummy_prompt)

        # Assert
        # We expect the model to identify the color blue.
        self.assertIn("blue", result.lower())
