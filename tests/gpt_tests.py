import unittest
from unittest.mock import patch, MagicMock, ANY
from PIL import Image
import torch
from .prompt_utils import _gpt_huggingface

# As the test file is in the same package, use a relative import.

class TestGptHuggingface(unittest.TestCase):
    """
    Tests for the _gpt_huggingface function.
    """

    @patch('torch.cuda.empty_cache')
    @patch('LatentUnfold.utils.prompt_utils.client', new_callable=MagicMock)
    def test_gpt_huggingface_flow(self, mock_client, mock_empty_cache):
        """
        Test the end-to-end flow of the _gpt_huggingface function,
        mocking the client and its components (processor, model).
        """
        # Arrange
        # 1. Create dummy inputs
        dummy_image = Image.new('RGB', (60, 30), color='red')
        dummy_prompt = "Describe the image."
        expected_description = "A description of a red image."

        # 2. Configure the mock client and its components
        mock_processor = MagicMock()
        mock_model = MagicMock()
        
        # The processor returns mock inputs for the model
        mock_inputs = {"input_ids": torch.tensor([[1, 2, 3]]), "pixel_values": torch.rand(1, 3, 30, 60)}
        mock_processor.return_value.to.return_value = mock_inputs
        
        # The model returns mock output tensors
        mock_output_tensors = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.generate.return_value = mock_output_tensors
        mock_model.device = 'cuda'
        
        # The processor decodes the model output to the final string
        mock_processor.batch_decode.return_value = [expected_description]

        # Assign the mocks to the client dictionary
        mock_client.__getitem__.side_effect = lambda key: {'processor': mock_processor, 'model': mock_model}[key]

        # Act
        result = _gpt_huggingface(dummy_image, dummy_prompt)

        # Assert
        # 1. Check that CUDA cache was cleared
        mock_empty_cache.assert_called_once()

        # 2. Check that the processor was called correctly
        mock_processor.assert_called_once_with(
            text=dummy_prompt,
            images=dummy_image,
            return_tensors="pt"
        )
        mock_processor.return_value.to.assert_called_once_with('cuda')

        # 3. Check that the model's generate method was called with the correct arguments
        mock_model.generate.assert_called_once()
        call_args, call_kwargs = mock_model.generate.call_args
        self.assertEqual(call_kwargs['max_new_tokens'], 512)
        self.assertEqual(call_kwargs['temperature'], 0.0)
        self.assertEqual(call_kwargs['do_sample'], False)
        self.assertEqual(call_kwargs['input_ids'], mock_inputs['input_ids'])
        self.assertEqual(call_kwargs['pixel_values'], mock_inputs['pixel_values'])


        # 4. Check that the processor's batch_decode was called with the model's output
        mock_processor.batch_decode.assert_called_once_with(mock_output_tensors, skip_special_tokens=True)

        # 5. Check that the function returned the expected final description
        self.assertEqual(result, expected_description)

if __name__ == '__main__':
    unittest.main()