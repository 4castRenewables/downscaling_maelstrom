#!/usr/bin/env bash
#
# __authors__ = Michael Langguth
# __date__  = '2022-01-21'
# __update__= '2022-02-28'
#
# **************** Description ****************
# This script can be used for setting up the virtual environment needed for downscaling with the U-net architecture
# as being implemented for the Tier-1 dataset in MAELSTROM (see https://www.maelstrom-eurohpc.eu/content/docs/uploads/doc6.pdf) 
# **************** Description ****************
#
### auxiliary-function S ###
check_argin() {
# Handle input arguments and check if one is equal to -lcontainer (not needed currently)
# Can also be used to check for non-positional arguments (such as -exp_id=*, see commented lines)
# !!! NOT USED YET!!!
    for argin in "$@"; do
        # if [[ $argin == *"-exp_id="* ]]; then
        #  exp_id=${argin#"-exp_id="}
        if [[ $argin == *"-lcontainer"* ]]; then
	        bool_container=1
        fi  
    done
    if [[ -z "${bool_container}" ]]; then
        bool_container=0
    fi
}
### auxiliary-function E ###

### MAIN S ###
#set -eu              # enforce abortion if a command is not re

SCR_SETUP="%create_env.sh: "

## some first sanity checks
# script is sourced?
if [[ ${BASH_SOURCE[0]} == "${0}" ]]; then
  echo "${SCR_SETUP}ERROR: 'create_env.sh' must be sourced, i.e. execute by prompting 'source create_env.sh [virt_env_name]'"
  exit 1
fi


# from now on, just return if something unexpected occurs instead of exiting
# as the latter would close the terminal including logging out
if [[ -z "$1" ]]; then
  echo "${SCR_SETUP}ERROR: Provide a name to set up the virtual environment, i.e. execute by prompting 'source create_env.sh [virt_env_name]"
  return
fi

# set some variables
HOST_NAME=$(hostname)
ENV_NAME=$1
SETUP_DIR=$(pwd)
SETUP_DIR_NAME="$(basename "${SETUP_DIR}")"
BASE_DIR="$(dirname "${SETUP_DIR}")"
VENV_DIR="${BASE_DIR}/virtual_envs/${ENV_NAME}"

## perform sanity checks
# * check if script is called from env_setup-directory
# * check if virtual env has already been set up

# script is called from env_setup-directory?
if [[ "${SETUP_DIR_NAME}" != "env_setup"  ]]; then
  echo "${SCR_SETUP}ERROR: Execute 'create_env.sh' from the env_setup-subdirectory only!"
  echo "${SETUP_DIR_NAME}"
  return
fi

# virtual environment already set-up?
if [[ -d ${VENV_DIR} ]]; then
  echo "${SCR_SETUP}Virtual environment has already been set up under ${VENV_DIR} and is ready to use."
  echo "NOTE: If you wish to set up a new virtual environment, delete the existing one or provide a different name."
  ENV_EXIST=1
else
  ENV_EXIST=0
fi

## check integratability of operating system
if [[ "${HOST_NAME}" == hdfml* || "${HOST_NAME}" == *jwlogin* ]]; then
  # unset PYTHONPATH to ensure that system-realted paths are not set
  unset PYTHONPATH
  modules_file="modules_jsc.sh"
else
  echo "${SCR_SETUP}ERROR: Model only runs on HDF-ML and Juwels (Booster) so far."
  return
fi

## set up virtual environment
if [[ "$ENV_EXIST" == 0 ]]; then
  # Install virtualenv-package and set-up virtual environment with required additional Python packages.
  echo "${SCR_SETUP}Configuring and activating virtual environment on ${HOST_NAME}"

  source "${modules_file}"

  python3 -m venv --system-site-packages "${VENV_DIR}"

  activate_virt_env=${VENV_DIR}/bin/activate

  echo "${SCR_SETUP}Entering virtual environment ${VENV_DIR} to install required Python modules..."
  source "${activate_virt_env}"
 
  # handle systematic issues with Stages/2022 
  MACHINE=$(hostname -f | cut -d. -f2)
  if [[ "${HOST}" == jwlogin2[2-4] ]]; then
     MACHINE="juwelsbooster"
  fi
  PY_VERSION=$(python --version 2>&1 | cut -d ' ' -f2 | cut -d. -f1-2)

  echo "${SCR_SETUP}Appending PYTHONPATH on ${MACHINE} for Python version ${PY_VERSION} to ensure proper set-up..."

  req_file=${SETUP_DIR}/requirements.txt

  # Without the environmental variables set above, we need to install wheel and explictly set the target directory
  pip3 install --no-cache-dir -r "${req_file}"

  # expand PYTHONPATH
  export PYTHONPATH=${BASE_DIR}:$PYTHONPATH >> "${activate_virt_env}"
  export PYTHONPATH=${BASE_DIR}/utils:$PYTHONPATH >> "${activate_virt_env}"
  export PYTHONPATH=${BASE_DIR}/handle_data:$PYTHONPATH >> "${activate_virt_env}"
  export PYTHONPATH=${BASE_DIR}/models:$PYTHONPATH >> "${activate_virt_env}"
  export PYTHONPATH=${BASE_DIR}/postprocess:$PYTHONPATH >> "${activate_virt_env}"
  export PYTHONPATH=${BASE_DIR}/preprocess:$PYTHONPATH >> "${activate_virt_env}"

  # ...and ensure that this also done when the
  echo "" >> "${activate_virt_env}"
  echo "# Expand PYTHONPATH..." >> "${activate_virt_env}"
  echo "export PYTHONPATH=${BASE_DIR}:\$PYTHONPATH" >> "${activate_virt_env}"
  echo "export PYTHONPATH=${BASE_DIR}/utils/:\$PYTHONPATH" >> "${activate_virt_env}"
  echo "export PYTHONPATH=${BASE_DIR}/models:\$PYTHONPATH " >> "${activate_virt_env}"
  echo "export PYTHONPATH=${BASE_DIR}/handle_data:\$PYTHONPATH" >> "${activate_virt_env}"
  echo "export PYTHONPATH=${BASE_DIR}/postprocess:\$PYTHONPATH" >> "${activate_virt_env}"
  echo "export PYTHONPATH=${BASE_DIR}/preprocess:\$PYTHONPATH" >> "${activate_virt_env}"

  info_str="Virtual environment ${VENV_DIR} has been set up successfully."
elif [[ "$ENV_EXIST" == 1 ]]; then
  # simply activate virtual environment
  info_str="Virtual environment ${VENV_DIR} has already been set up before. Nothing to be done."
fi

echo "${SCR_SETUP}${info_str}"
### MAIN E ###
