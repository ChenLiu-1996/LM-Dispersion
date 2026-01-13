# Transformer Dispersion

## Usage
1. Mid-train GPT2
    Under `transformer_dispersion/midtrain_gpt2_huggingface`
    ```
    accelerate launch midtrain_gpt2.py --train_tokens 300_000_000 --dispersion 'Covariance' --dispersion_loc 'all' --dispersion_coeff 1.0 --hf_token $HUGGINGFACE_ACCESS_TOKEN
    ```


### Visualize observations
```
python plot_trend.py --model-id openai-community-gpt2 openai-community-gpt2-medium openai-community-gpt2-large openai-community-gpt2-xl --model-family gpt2
python plot_trend.py --model-id Qwen-Qwen-1_8B Qwen-Qwen-7B Qwen-Qwen-14B Qwen-Qwen-72B --model-family Qwen1
python plot_trend.py --model-id Qwen-Qwen2.5-0.5B Qwen-Qwen2.5-1.5B Qwen-Qwen2.5-3B Qwen-Qwen2.5-7B Qwen-Qwen2.5-14B Qwen-Qwen2.5-32B Qwen-Qwen2.5-72B --model-family Qwen2.5
python plot_trend.py --model-id Qwen-Qwen3-0.6B Qwen-Qwen3-1.7B Qwen-Qwen3-4B Qwen-Qwen3-8B Qwen-Qwen3-14B Qwen-Qwen3-32B --model-family Qwen3
python plot_trend.py --model-id bigscience-bloom-560m bigscience-bloom-1b1 bigscience-bloom-1b7 bigscience-bloom-3b bigscience-bloom-7b1 --model-family bloom
python plot_trend.py --paired --model-id Qwen-Qwen2.5-Math-1.5B Qwen-Qwen2.5-Math-7B Qwen-Qwen2.5-14B Qwen-Qwen2.5-32B deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B deepseek-ai-DeepSeek-R1-Distill-Qwen-7B deepseek-ai-DeepSeek-R1-Distill-Qwen-14B deepseek-ai-DeepSeek-R1-Distill-Qwen-32B --model-family Qwen2.5-distill
```

## Dependencies
We developed the codebase in a miniconda environment.
How we created the conda environment:
```
# Optional: Update to libmamba solver.
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

conda create --name transformer pytorch==2.1.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -c anaconda -c conda-forge -y
conda activate transformer
conda install scikit-learn scikit-image pandas matplotlib seaborn tqdm -c pytorch -c anaconda -c conda-forge -y

python -m pip install webdataset einops open-clip-torch
python -m pip install git+https://github.com/openai/CLIP.git
python -m pip install diffusers["torch"]==0.21.4 transformers huggingface_hub==0.25.2
python -m pip install datasets sentencepiece
python -m pip install numpy==1.26
python -m pip install nltk

python -m pip install -U phate
python -m pip install trl bitsandbytes
python -m pip install "transformers==4.46.0"
python -m pip install -U transformers accelerate
python -m pip install lm-eval
```