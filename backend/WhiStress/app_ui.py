import torch
import gradio as gr
from pathlib import Path
from whistress import WhiStressInferenceClient

CURRENT_DIR = Path(__file__).parent
# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhiStressInferenceClient(device=device)


def get_whistress_predictions(audio):
    """
    Get the transcription and emphasis scores for the given audio input.
    Args:
        audio (sr, numpy.ndarray): The audio input as a NumPy array.
    Returns:
        List[Tuple[str, int]]: A list of tuples containing words and their emphasis scores.
    """
    audio = {
        "sampling_rate": audio[0],
        "array": audio[1],
    } 
    return model.predict(audio=audio, transcription=None, return_pairs=True)


# App UI
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown(
                """
                # ***WhiStress***: Enriching Transcriptions with Sentence Stress Detection
                WhiStress allows you to detect emphasized words in your speech.
                
                Check out our paper: 📚 [***WhiStress***](https://arxiv.org/)
                
                ## Architecture
                The model is built on [Whisper](https://arxiv.org/abs/2212.04356) model,
                using `whisper-small.en` [model](https://huggingface.co/openai/whisper-small.en)
                as the backbone.
                WhiStress includes an additional decoder based classifier that predicts the stress label of each transcription token.
                
                ## Training Data
                WhiStress was trained using [***TinyStress-15K***](https://huggingface.co/datasets/slprl/TinyStress-15K),
                that is derived from the [tinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.
                
                ## Inference Demo
                Upload an audio file or record your own voice to transcribe the speech and emphasize the important words.
                
                For maximal performance, please speak clearly.
                """
            )
        with gr.Column(scale=1):
            # Define Gradio interface for displaying image with HTML component
            gr.Image(
                f"{CURRENT_DIR}/assets/whistress_model.svg",
                label="Architecture",
            )

    gr.Interface(
        get_whistress_predictions,
        gr.Audio(
                sources=["microphone", "upload"],
                label="Upload speech or record your own",
                type="numpy",
            ),
        gr.HighlightedText(),
        allow_flagging="never",
    )


def launch():
    demo.launch()


if __name__ == "__main__":
    launch()
