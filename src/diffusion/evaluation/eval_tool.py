import os
from typing import Optional

import streamlit as st
import torch
from diffusers import StableDiffusionPipeline


def create_pipeline(base_model_path: str, lora_weights_path: str):
    # Check if pipeline already exists in session state
    if (
        "pipe" not in st.session_state
        or st.session_state.current_checkpoint != lora_weights_path
    ):
        # Load base model
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        ).to("cuda")

        # Load and fuse LoRA weights from local checkpoint
        pipe.load_lora_weights(lora_weights_path)

        # Store in session state
        st.session_state.pipe = pipe
        st.session_state.current_checkpoint = lora_weights_path

    return st.session_state.pipe


def generate_images(
    pipe,
    prompt: str,
    num_images: int = 5,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    height: int = 512,
    width: int = 512,
    seed: Optional[int] = None,
):
    if seed is not None:
        generator = torch.Generator("cuda").manual_seed(seed)
    else:
        generator = None

    images = []
    for _ in range(num_images):
        image = pipe(
            prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        images.append(image)
        if seed is not None:
            seed += 1  # Increment seed for variety
    return images


def main() -> None:
    st.set_page_config(page_title="Shape Generator", layout="wide")

    # Initialize session state for model
    if "pipe" not in st.session_state:
        st.session_state.pipe = None
        st.session_state.current_checkpoint = None

    # Title with custom styling
    st.markdown(
        """
        <h1 style='text-align: center; color: #1E88E5;'>
            🎨 AI Shape Generator
        </h1>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar for model settings
    with st.sidebar:
        st.markdown("## 🛠️ Model Settings")

        # Add checkpoint selector
        st.markdown("### 📂 Model Checkpoint")
        checkpoint_dir = "sd-shapes-model"
        if os.path.exists(checkpoint_dir):
            checkpoints = [
                d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")
            ]
            if checkpoints:
                selected_checkpoint = st.selectbox(
                    "Select checkpoint",
                    options=checkpoints,
                    format_func=lambda x: f"Checkpoint {x.split('-')[1]}",
                )
                checkpoint_path = os.path.join(checkpoint_dir, selected_checkpoint)
            else:
                st.error("No checkpoints found in directory")
                st.stop()
        else:
            st.error(f"Directory {checkpoint_dir} not found")
            st.stop()

        # Advanced Settings
        st.markdown("### ⚙️ Generation Settings")

        guidance_scale = st.slider(
            "Guidance Scale",
            min_value=1.0,
            max_value=20.0,
            value=7.5,
            help="Higher values make images more closely match the prompt",
        )

        num_inference_steps = st.slider(
            "Inference Steps",
            min_value=20,
            max_value=150,
            value=50,
            help="More steps = higher quality but slower generation",
        )

        col1, col2 = st.columns(2)
        with col1:
            height = st.select_slider(
                "Height",
                options=[128, 256, 512, 768, 1024],
                value=512,
                help="Image height in pixels",
            )
        with col2:
            width = st.select_slider(
                "Width",
                options=[128, 256, 512, 768, 1024],
                value=512,
                help="Image width in pixels",
            )

        seed_input = st.number_input(
            "Random Seed",
            value=-1,
            help="Set a seed for reproducible images. -1 for random",
        )
        generator_seed: Optional[int] = None if seed_input == -1 else seed_input

        num_images = st.slider(
            "Number of Images",
            1,
            9,
            5,
            help="Number of images to generate",
        )

    # Main content
    st.markdown("### 📝 Enter your prompt")
    prompt = st.text_area(
        "",
        value="Triangle with color PURPLE is at distance 7.5 and centered at (303, 153)",
        height=100,
    )

    if st.button("🎨 Generate Images", type="primary"):
        try:
            with st.spinner("🎨 Creating your masterpiece..."):
                # Pipeline will only be loaded if it's not in session state
                # or if checkpoint has changed
                pipe = create_pipeline(
                    "runwayml/stable-diffusion-v1-5",
                    checkpoint_path,
                )

                images = generate_images(
                    pipe=pipe,
                    prompt=prompt,
                    num_images=num_images,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    height=height,
                    width=width,
                    seed=generator_seed,
                )

                # Display images in a grid
                cols = st.columns(3)
                for idx, image in enumerate(images):
                    with cols[idx % 3]:
                        st.image(image, caption=f"Generation {idx + 1}")

        except Exception as e:
            st.error(f"Error generating images: {str(e)}")

    # Add some helpful information
    with st.expander("ℹ️ Tips for better results"):
        st.markdown(
            """
        - Be specific about shapes, colors, and arrangements
        - Try different guidance scales
        - Experiment with different numbers of shapes
        - Use clear spatial relationships (e.g., "above", "next to")
        """
        )


if __name__ == "__main__":
    main()
