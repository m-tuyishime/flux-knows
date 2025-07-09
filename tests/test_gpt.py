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
