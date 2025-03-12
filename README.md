# TipToi Translation (t3)

## Usage
- Once you've followed the setup instructions below you can run the application
- Run the following from repository's root
  - `python -m t3 'Rekorde im Tierreich.gme' workdir`
  - This will translate `Rekorde im Tierreich.gme` and store the translated GME and intermediate files in the `workdir`
  - Run `python -m t3 -h` to see all available options
- Alternatively run the application from a Docker container (see instructions below)

## Development

### Manual setup
- Clone this repo with submodules: `git clone --recurse-submodules git@github.com:jtomori/t3.git`
- `sudo apt install sox ffmpeg`
- `pip install numpy typing_extensions`
- `pip install -r requirements.txt`
- Store [SeamlessExpressive](https://huggingface.co/facebook/seamless-expressive) models in the `SeamlessExpressive` folder in repository's root
- Compile `libtiptoi.c`: `gcc tip-toi-reveng/libtiptoi.c -o libtiptoi`

### Docker
- GPU inference requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Build image with `docker build -t t3 .`
- Run container with `docker run --runtime=nvidia --gpus all --volume ./SeamlessExpressive:/app/SeamlessExpressive --volume ./gme:/app/gme --volume ./workdir:/app/workdir --rm --name t3 t3 gme/name_of_file.gme workdir`
  - Make sure that `gme, SeamlessExpressive, workdir` directories are present in your current directory
  - `workdir` will contain translated GME file along with intermediate files, CSV report
  - Omit `--runtime=nvidia --gpus all` for performing a CPU inference

### Tests & code checks
- `python tests.py`
- `./checks.sh`

## Releases

### v1.1 - 2025-03-12
- Finished setup for running GPU (or CPU) inference from a Docker container

### v1.0 - 2025-03-11
- Initial release
