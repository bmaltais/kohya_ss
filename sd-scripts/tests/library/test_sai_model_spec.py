"""Tests for sai_model_spec module."""

import pytest
import time

from library import sai_model_spec


class MockArgs:
    """Mock argparse.Namespace for testing."""

    def __init__(self, **kwargs):
        # Default values
        self.v2 = False
        self.v_parameterization = False
        self.resolution = 512
        self.metadata_title = None
        self.metadata_author = None
        self.metadata_description = None
        self.metadata_license = None
        self.metadata_tags = None
        self.min_timestep = None
        self.max_timestep = None
        self.clip_skip = None
        self.output_name = "test_output"

        # Override with provided values
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestModelSpecMetadata:
    """Test the ModelSpecMetadata dataclass."""

    def test_creation_and_conversion(self):
        """Test creating dataclass and converting to metadata dict."""
        metadata = sai_model_spec.ModelSpecMetadata(
            architecture="stable-diffusion-v1",
            implementation="diffusers",
            title="Test Model",
            resolution="512x512",
            author="Test Author",
            description=None,  # Test None exclusion
        )

        assert metadata.architecture == "stable-diffusion-v1"
        assert metadata.sai_model_spec == "1.0.1"

        metadata_dict = metadata.to_metadata_dict()
        assert "modelspec.architecture" in metadata_dict
        assert "modelspec.author" in metadata_dict
        assert "modelspec.description" not in metadata_dict  # None values excluded
        assert metadata_dict["modelspec.sai_model_spec"] == "1.0.1"

    def test_additional_fields_handling(self):
        """Test handling of additional metadata fields."""
        additional = {"custom_field": "custom_value", "modelspec.prefixed": "prefixed_value"}

        metadata = sai_model_spec.ModelSpecMetadata(
            architecture="stable-diffusion-v1",
            implementation="diffusers",
            title="Test Model",
            resolution="512x512",
            additional_fields=additional,
        )

        metadata_dict = metadata.to_metadata_dict()
        assert "modelspec.custom_field" in metadata_dict
        assert "modelspec.prefixed" in metadata_dict
        assert metadata_dict["modelspec.custom_field"] == "custom_value"

    def test_from_args_extraction(self):
        """Test creating ModelSpecMetadata from args with metadata_* fields."""
        args = MockArgs(metadata_author="Test Author", metadata_trigger_phrase="anime style", metadata_usage_hint="Use CFG 7.5")

        metadata = sai_model_spec.ModelSpecMetadata.from_args(
            args,
            architecture="stable-diffusion-v1",
            implementation="diffusers",
            title="Test Model",
            resolution="512x512",
        )

        assert metadata.author == "Test Author"
        assert metadata.additional_fields["trigger_phrase"] == "anime style"
        assert metadata.additional_fields["usage_hint"] == "Use CFG 7.5"


class TestArchitectureDetection:
    """Test architecture detection for different model types."""

    @pytest.mark.parametrize(
        "config,expected",
        [
            ({"v2": False, "v_parameterization": False, "sdxl": True}, "stable-diffusion-xl-v1-base"),
            ({"v2": False, "v_parameterization": False, "sdxl": False, "model_config": {"flux": "dev"}}, "flux-1-dev"),
            ({"v2": False, "v_parameterization": False, "sdxl": False, "model_config": {"flux": "chroma"}}, "chroma"),
            (
                {"v2": False, "v_parameterization": False, "sdxl": False, "model_config": {"sd3": "large"}},
                "stable-diffusion-3-large",
            ),
            ({"v2": True, "v_parameterization": True, "sdxl": False}, "stable-diffusion-v2-768-v"),
            ({"v2": False, "v_parameterization": False, "sdxl": False}, "stable-diffusion-v1"),
        ],
    )
    def test_architecture_detection(self, config, expected):
        """Test architecture detection for various model configurations."""
        model_config = config.pop("model_config", None)
        arch = sai_model_spec.determine_architecture(lora=False, textual_inversion=False, model_config=model_config, **config)
        assert arch == expected

    def test_adapter_suffixes(self):
        """Test LoRA and textual inversion suffixes."""
        lora_arch = sai_model_spec.determine_architecture(
            v2=False, v_parameterization=False, sdxl=True, lora=True, textual_inversion=False
        )
        assert lora_arch == "stable-diffusion-xl-v1-base/lora"

        ti_arch = sai_model_spec.determine_architecture(
            v2=False, v_parameterization=False, sdxl=False, lora=False, textual_inversion=True
        )
        assert ti_arch == "stable-diffusion-v1/textual-inversion"


class TestImplementationDetection:
    """Test implementation detection for different model types."""

    @pytest.mark.parametrize(
        "config,expected",
        [
            ({"model_config": {"flux": "dev"}}, "https://github.com/black-forest-labs/flux"),
            ({"model_config": {"flux": "chroma"}}, "https://huggingface.co/lodestones/Chroma"),
            ({"model_config": {"lumina": "lumina2"}}, "https://github.com/Alpha-VLLM/Lumina-Image-2.0"),
            ({"lora": True, "sdxl": True}, "https://github.com/Stability-AI/generative-models"),
            ({"lora": True, "sdxl": False}, "diffusers"),
        ],
    )
    def test_implementation_detection(self, config, expected):
        """Test implementation detection for various configurations."""
        model_config = config.pop("model_config", None)
        impl = sai_model_spec.determine_implementation(
            lora=config.get("lora", False), textual_inversion=False, sdxl=config.get("sdxl", False), model_config=model_config
        )
        assert impl == expected


class TestResolutionHandling:
    """Test resolution parsing and defaults."""

    @pytest.mark.parametrize(
        "input_reso,expected",
        [
            ((768, 1024), "768x1024"),
            (768, "768x768"),
            ("768,1024", "768x1024"),
        ],
    )
    def test_explicit_resolution_formats(self, input_reso, expected):
        """Test different resolution input formats."""
        res = sai_model_spec.determine_resolution(reso=input_reso)
        assert res == expected

    @pytest.mark.parametrize(
        "config,expected",
        [
            ({"sdxl": True}, "1024x1024"),
            ({"model_config": {"flux": "dev"}}, "1024x1024"),
            ({"v2": True, "v_parameterization": True}, "768x768"),
            ({}, "512x512"),  # Default SD v1
        ],
    )
    def test_default_resolutions(self, config, expected):
        """Test default resolution detection."""
        model_config = config.pop("model_config", None)
        res = sai_model_spec.determine_resolution(model_config=model_config, **config)
        assert res == expected


class TestThumbnailProcessing:
    """Test thumbnail data URL processing."""

    def test_file_to_data_url(self):
        """Test converting file to data URL."""
        import tempfile
        import os

        # Create a tiny test PNG (1x1 pixel)
        test_png_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xff\xff\xff\x00\x00\x00\x04\x00\x01\x9d\xb3\xa7c\x00\x00\x00\x00IEND\xaeB`\x82"

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(test_png_data)
            temp_path = f.name

        try:
            data_url = sai_model_spec.file_to_data_url(temp_path)

            # Check format
            assert data_url.startswith("data:image/png;base64,")

            # Check it's a reasonable length (base64 encoded)
            assert len(data_url) > 50

            # Verify we can decode it back
            import base64

            encoded_part = data_url.split(",", 1)[1]
            decoded_data = base64.b64decode(encoded_part)
            assert decoded_data == test_png_data

        finally:
            os.unlink(temp_path)

    def test_file_to_data_url_nonexistent_file(self):
        """Test error handling for nonexistent files."""
        import pytest

        with pytest.raises(FileNotFoundError):
            sai_model_spec.file_to_data_url("/nonexistent/file.png")

    def test_thumbnail_processing_in_metadata(self):
        """Test thumbnail processing in build_metadata_dataclass."""
        import tempfile
        import os

        # Create a test image file
        test_png_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xff\xff\xff\x00\x00\x00\x04\x00\x01\x9d\xb3\xa7c\x00\x00\x00\x00IEND\xaeB`\x82"

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(test_png_data)
            temp_path = f.name

        try:
            timestamp = time.time()

            # Test with file path - should be converted to data URL
            metadata = sai_model_spec.build_metadata_dataclass(
                state_dict=None,
                v2=False,
                v_parameterization=False,
                sdxl=False,
                lora=False,
                textual_inversion=False,
                timestamp=timestamp,
                title="Test Model",
                optional_metadata={"thumbnail": temp_path},
            )

            # Should be converted to data URL
            assert "thumbnail" in metadata.additional_fields
            assert metadata.additional_fields["thumbnail"].startswith("data:image/png;base64,")

        finally:
            os.unlink(temp_path)

    def test_thumbnail_data_url_passthrough(self):
        """Test that existing data URLs are passed through unchanged."""
        timestamp = time.time()

        existing_data_url = (
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )

        metadata = sai_model_spec.build_metadata_dataclass(
            state_dict=None,
            v2=False,
            v_parameterization=False,
            sdxl=False,
            lora=False,
            textual_inversion=False,
            timestamp=timestamp,
            title="Test Model",
            optional_metadata={"thumbnail": existing_data_url},
        )

        # Should be unchanged
        assert metadata.additional_fields["thumbnail"] == existing_data_url

    def test_thumbnail_invalid_file_handling(self):
        """Test graceful handling of invalid thumbnail files."""
        timestamp = time.time()

        metadata = sai_model_spec.build_metadata_dataclass(
            state_dict=None,
            v2=False,
            v_parameterization=False,
            sdxl=False,
            lora=False,
            textual_inversion=False,
            timestamp=timestamp,
            title="Test Model",
            optional_metadata={"thumbnail": "/nonexistent/file.png"},
        )

        # Should be removed from additional_fields due to error
        assert "thumbnail" not in metadata.additional_fields


class TestBuildMetadataIntegration:
    """Test the complete metadata building workflow."""

    def test_sdxl_model_workflow(self):
        """Test complete workflow for SDXL model."""
        timestamp = time.time()

        metadata = sai_model_spec.build_metadata_dataclass(
            state_dict=None,
            v2=False,
            v_parameterization=False,
            sdxl=True,
            lora=False,
            textual_inversion=False,
            timestamp=timestamp,
            title="Test SDXL Model",
        )

        assert metadata.architecture == "stable-diffusion-xl-v1-base"
        assert metadata.implementation == "https://github.com/Stability-AI/generative-models"
        assert metadata.resolution == "1024x1024"
        assert metadata.prediction_type == "epsilon"

    def test_flux_model_workflow(self):
        """Test complete workflow for Flux model."""
        timestamp = time.time()

        metadata = sai_model_spec.build_metadata_dataclass(
            state_dict=None,
            v2=False,
            v_parameterization=False,
            sdxl=False,
            lora=False,
            textual_inversion=False,
            timestamp=timestamp,
            title="Test Flux Model",
            model_config={"flux": "dev"},
            optional_metadata={"trigger_phrase": "anime style"},
        )

        assert metadata.architecture == "flux-1-dev"
        assert metadata.implementation == "https://github.com/black-forest-labs/flux"
        assert metadata.prediction_type is None  # Flux doesn't use prediction_type
        assert metadata.additional_fields["trigger_phrase"] == "anime style"

    def test_legacy_function_compatibility(self):
        """Test that legacy build_metadata function works correctly."""
        timestamp = time.time()

        metadata_dict = sai_model_spec.build_metadata(
            state_dict=None,
            v2=False,
            v_parameterization=False,
            sdxl=True,
            lora=False,
            textual_inversion=False,
            timestamp=timestamp,
            title="Test Model",
        )

        assert isinstance(metadata_dict, dict)
        assert metadata_dict["modelspec.sai_model_spec"] == "1.0.1"
        assert metadata_dict["modelspec.architecture"] == "stable-diffusion-xl-v1-base"
