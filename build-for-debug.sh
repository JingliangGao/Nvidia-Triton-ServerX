#!/bin/bash

# set variables
CURRENT_DIR=$(pwd)
BUILD_DIR="build"

# clear the terminal
clear

# check if sudoers
if [ "$EUID" -ne 0 ]; then
    SUDO_ER='sudo'
else
    SUDO_ER=''
fi

# refresh build folder
echo "[INFO] Refresh build folder ... "
cd ${CURRENT_DIR}
if [ -d ${BUILD_DIR} ] ; then
    ${SUDO_ER} rm -rf ${BUILD_DIR}
fi
mkdir ${BUILD_DIR}

# build project 
cd ${CURRENT_DIR}/${BUILD_DIR}
cmake ..
make -j

echo "[INFO] All done!"