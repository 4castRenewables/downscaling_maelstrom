In juwelsbooster
1. Set python to version 3.9. For this load the following modules:
```
ml --force purge
ml use $OTHERSTAGES
ml Stages/2022
ml GCCcore/.11.2.0
ml Python/3.9.6
```

2. Create a virtual enviroment and activate it:
```
python -m venv <venv-name>
source <venv-name>/bin/activate
```

3. Install ap5 dependencies with pip. The requirements file is in the `env_setup` file
```
pip install -r downscaling_ap5/env_setup/requirements_wo_modules.txt
```

4. Add the following code at the end of the virtual enviroment activate file `<venv-name>/bin/activate`:
```
BASE_DIR="<absolute path to downscaling_maelstrom/downscaling_ap5 directory>"
# expand PYTHONPATH
export PYTHONPATH=${BASE_DIR}:$PYTHONPATH
export PYTHONPATH=${BASE_DIR}/utils:$PYTHONPATH
export PYTHONPATH=${BASE_DIR}/handle_data:$PYTHONPATH
export PYTHONPATH=${BASE_DIR}/models:$PYTHONPATH
export PYTHONPATH=${BASE_DIR}/postprocess:$PYTHONPATH
export PYTHONPATH=${BASE_DIR}/preprocess:$PYTHONPATH
```
<br>
In your local mlproject

1. Set `Python` in `unicore-config-venv.yaml` to the path of your virtual enviroment

```
Environment:
  Python: /path/to/<venv-name>
```

2. Run your experiment with mantik
```
mantik runs submit <absolute path to downscaling_ap5/mlproject directory> --backend-config unicore-config-venv.yaml --entry-point main --experiment-id 59 -v
```