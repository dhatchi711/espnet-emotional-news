# Setup

This README provides step-by-step instructions for setting up and running the emotional news speech synthesis model using ESPnet on a remote server and accessing it from your local machine.

## Remote Server Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/espnet/espnet.git
   cd espnet/tools
   ```

1. Set up the Conda environment:
   * If you already have Conda installed, create a new environment for ESPnet:
     ```bash
     ./setup_anaconda.sh ${CONDA_ROOT} espnet 3.10
     ```
     > Note: Replace `${CONDA_ROOT}` with your Conda root directory (you can find it with `conda env list`).

1. Activate the ESPnet environment:
   ```bash
   conda activate espnet
   ```

1. Install ESPnet dependencies:
   * Ensure NVCC is available (you may need to start an SRUN session with GPU)
   * Run the following command to install dependencies:
     ```bash
     make TH_VERSION=2.4
     ```
   * The CUDA version will be automatically selected based on `nvcc --version`.
   * NOTE: Remove lightning.done from your MAKEFILE since it gives some installaiton issues due to version mismatches. If you are running on the current branch then it is already removed.

1. Install other packages needed for running app
   * `pip install gradio`
   * `pip install transformers`
   * `pip install webrtcvad`
   * `cd espnet/tools && ./installers/install_whisper.sh`

1. Create a Hugging Face token:
   * Follow the instructions at: https://huggingface.co/docs/hub/en/security-tokens
   * Go to HuggingFace settings and create a new token
   * Copy the token to a secure location

1. Set your Hugging Face token as an environment variable:
   ```bash
   export HF_TOKEN=your_token_here
   ```

1. Navigate to the emotional news speech synthesis directory and start gradio server:
   ```bash
   cd ../egs2/emo_news_sds/sds1
   ./run.sh
   ```
   * Ensure you run `./run.sh` on an srun session with CUDA gpus else it will throw errors.

## Local Machine Setup

1. Forward the SSH port to access the web application:
   ```bash
   ssh -L local_port:localhost:remote_port username@remote_server_address
   ```
   > Note: Replace `local_port`, `remote_port`, `username`, and `remote_server_address` with your specific values. The remote port will be shown in the output when the Gradio server starts.

2. Access the web application:
   * Open your web browser
   * Navigate to: `http://localhost:local_port`
   * Replace `local_port` with the port number you specified in the SSH command

## Troubleshooting

* If NVCC is not found, ensure you are in a GPU-enabled session
* If you encounter Conda environment issues, verify your Conda root directory is correct
* For Gradio server connection problems, check that your port forwarding is configured correctly

## Additional Resources

* ESPnet Documentation: https://espnet.github.io/espnet/
* Hugging Face Documentation: https://huggingface.co/docs