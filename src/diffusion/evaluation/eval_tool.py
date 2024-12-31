import streamlit as st
import torch

from src.diffusion.modules.diffusion_model import DiffusionModel


def main():
    st.title("Diffusion Model Evaluation")
    model = DiffusionModel.load_from_checkpoint("path/to/checkpoint.ckpt")
    st.write("Model loaded successfully!")

    input_noise = st.slider("Input Noise", 0, 100, 50)
    generated_image = model.generate(torch.randn((1, 1, 64, 64)) * input_noise)

    st.image(
        generated_image.squeeze().numpy(),
        caption="Generated Image",
        use_column_width=True,
    )


if __name__ == "__main__":
    main()
