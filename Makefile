#################################################################################
# GLOBALS                                                                       #
#################################################################################
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = icairdpath

# docker options
DOCKER_IMAGE_NAME = icairdpath
DOCKER_USER_NAME = ubuntu

# python options
VENV_NAME = icairdpath
PYTHON_INTERPRETER = python
PYTHON_VERSION = 3.6

# docker build process info
GIT_USER = davemor # CHANGE THIS
GIT_TOKEN = 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 # CHANGE THIS
DATA_WRITERS_GROUP_ID =  $(shell getent group icaird-data-writers | cut -d: -f3)

JUPYTER_PORT := 8280

#################################################################################
# PYTHON ENVIRONMENT COMMANDS                                                   #
#################################################################################

## set up the python environment
create_environment:
	conda create --name $(VENV_NAME) python=$(PYTHON_VERSION)
	conda init bash
	echo "source activate icairdpath" > ~/.bashrc
	export PATH=/opt/conda/envs/env/bin:$PATH

## install the requirements into the python environment
requirements: install_asap install_openslide install_isyntax_sdk
	conda env update --file environment.yml
	pip install -r requirements.txt

## save the python environment so it can be recreated
export_environment:
	conda env export --no-builds | grep -v "^prefix: " > environment.yml

#################################################################################
# OTHER DEPENDENCIES		                                                    #
#################################################################################
ASAP_LOCATION = https://github.com/computationalpathologygroup/ASAP/releases/download/1.9/ASAP-1.9-Linux-Ubuntu1804.deb
install_asap:
	sudo apt-get update
	curl -o ASAP-1.9-Linux-Ubuntu1804.deb -L $(ASAP_LOCATION)
	sudo apt -y install ./ASAP-1.9-Linux-Ubuntu1804.deb
	rm ASAP-1.9-Linux-Ubuntu1804.deb

install_openslide:
	sudo apt-get update
	sudo apt install -y build-essential
	sudo apt-get -y install openslide-tools
	pip install Pillow
	pip install openslide-python

install_isyntax_sdk:
	sudo apt install gdebi -y
	sudo gdebi -n ./libraries/philips-pathology-sdk/*pixelengine*.deb
	sudo gdebi -n ./libraries/philips-pathology-sdk/*eglrendercontext*.deb
	sudo gdebi -n ./libraries/philips-pathology-sdk/*gles2renderbackend*.deb
	sudo gdebi -n ./libraries/philips-pathology-sdk/*gles3renderbackend*.deb
	sudo gdebi -n ./libraries/philips-pathology-sdk/*softwarerenderer*.deb

install_java:
	sudo apt -y install software-properties-common
	sudo add-apt-repository ppa:webupd8team/java
	sudo apt -y install openjdk-8-jdk
	sudo update-alternatives --config java # select Java 8
	printf '\nexport JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64' >> ~/.bashrc
	export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

#################################################################################
# CONTAINER COMMANDS                                                            #
#################################################################################
docker_image:
	@echo $(DATA_WRITERS_GROUP_ID)
	docker build --build-arg GIT_USER=$(GIT_USER) \
				 --build-arg GIT_TOKEN=$(GIT_TOKEN) \
				 --build-arg DATA_WRITERS_GROUP_ID=$(DATA_WRITERS_GROUP_ID) \
				 -t $(DOCKER_IMAGE_NAME) .
 
docker_image_paige_endo:
	docker build \
				 --build-arg GIT_USER=$(GIT_USER) \
				 --build-arg GIT_TOKEN=$(GIT_TOKEN) \
				 --build-arg DATA_WRITERS_GROUP_ID=$(DATA_WRITERS_GROUP_ID) \
				 --file ./Dockerfile.endo \
				 -t icaird_paige_endo .

docker_image_paige_cerv:
	docker build \
				 --build-arg GIT_USER=$(GIT_USER) \
				 --build-arg GIT_TOKEN=$(GIT_TOKEN) \
				 --build-arg DATA_WRITERS_GROUP_ID=$(DATA_WRITERS_GROUP_ID) \
				 --file ./Dockerfile.cerv \
				 -t icaird_paige_cerv .				 

docker_run_paige_endo_test:
	docker run --shm-size=64G \
				--gpus all \
				--rm \
				-v "$(pwd)":/hostpwd \
				-v /home/david/icairdpath/repath:/home/$(DOCKER_USER_NAME)/$(PROJECT_NAME)/repath \
				-v /raid/datasets:/home/$(DOCKER_USER_NAME)/$(PROJECT_NAME)/data \
				-v /raid/experiments/repath:/home/$(DOCKER_USER_NAME)/$(PROJECT_NAME)/results \
				-v /raid/experiments/repath/final_models:/home/$(DOCKER_USER_NAME)/$(PROJECT_NAME)/models \
				-v /mnt/isilon1/:/home/$(DOCKER_USER_NAME)/$(PROJECT_NAME)/data/icaird \
				-v /mnt/isilon1/camelyon17/:/home/$(DOCKER_USER_NAME)/$(PROJECT_NAME)/data/camelyon17 \
				-it icaird_paige_endo \
				run-once \
				--manifest '{"inputs":[{"name":"slide","media_type":"wsi/vnd.paige.ai.wsi.syntax","url":"file:///home/ubuntu/icairdpath/data/icaird/iCAIRD/IC-EN-00007-01.isyntax","metadata":{"foo":"bar"}}],"results":[{"name":"prediction","url":"file:///home/ubuntu/icairdpath/results/paige_output_testing/test_output_pred.json"},{"name":"heatmap","url":"file:///home/ubuntu/icairdpath/results/paige_output_testing/test_output_tiff.tif"}]}'

docker_run_paige_cerv_test:
	docker run --shm-size=64G \
				--gpus all \
				--rm \
				-v "$(pwd)":/hostpwd \
				-v /home/david/icairdpath/repath:/home/$(DOCKER_USER_NAME)/$(PROJECT_NAME)/repath \
				-v /raid/datasets:/home/$(DOCKER_USER_NAME)/$(PROJECT_NAME)/data \
				-v /raid/experiments/repath:/home/$(DOCKER_USER_NAME)/$(PROJECT_NAME)/results \
				-v /raid/experiments/repath/final_models:/home/$(DOCKER_USER_NAME)/$(PROJECT_NAME)/models \
				-v /mnt/isilon1/:/home/$(DOCKER_USER_NAME)/$(PROJECT_NAME)/data/icaird \
				-v /mnt/isilon1/camelyon17/:/home/$(DOCKER_USER_NAME)/$(PROJECT_NAME)/data/camelyon17 \
				-it icaird_paige_cerv \
				run-once \
				--manifest '{"inputs":[{"name":"slide","media_type":"wsi/vnd.paige.ai.wsi.syntax","url":"file:///home/ubuntu/icairdpath/data/icaird/iCAIRD/IC-CX-00005-01.isyntax","metadata":{"foo":"bar"}}],"results":[{"name":"prediction","url":"file:///home/ubuntu/icairdpath/results/paige_output_testing/test_cervical_pred.json"},{"name":"mal_hm","url":"file:///home/ubuntu/icairdpath/results/paige_output_testing/test_mal.tif"},{"name":"high_hm","url":"file:///home/ubuntu/icairdpath/results/paige_output_testing/test_high.tif"},{"name":"low_hm","url":"file:///home/ubuntu/icairdpath/results/paige_output_testing/test_low.tif"},{"name":"norm_hm","url":"file:///home/ubuntu/icairdpath/results/paige_output_testing/test_norm.tif"}]}'

docker_run:
	docker run --shm-size=64G \
				--gpus all -p $(JUPYTER_PORT):$(JUPYTER_PORT) \
				-v /raid/datasets:/home/$(DOCKER_USER_NAME)/$(PROJECT_NAME)/data \
				-v /raid/experiments/repath:/home/$(DOCKER_USER_NAME)/$(PROJECT_NAME)/results \
				-v /raid/experiments/repath/final_models:/home/$(DOCKER_USER_NAME)/$(PROJECT_NAME)/models \
				-v /mnt/isilon1/:/home/$(DOCKER_USER_NAME)/$(PROJECT_NAME)/data/icaird \
				-v /mnt/isilon1/camelyon17/:/home/$(DOCKER_USER_NAME)/$(PROJECT_NAME)/data/camelyon17 \
				-it $(DOCKER_IMAGE_NAME):latest

docker_run_local:
	docker run --shm-size=16G --gpus all -p $(JUPYTER_PORT):$(JUPYTER_PORT) \
				-v $(PROJECT_DIR):/home/$(DOCKER_USER_NAME)/$(PROJECT_NAME) \
				-it $(PROJECT_NAME):latest


#################################################################################
# JUPYTER COMMANDS                                                              #
#################################################################################
setup_jupyter:
	pip install --user ipykernel
	python -m ipykernel install --user --name=icairdpath

run_notebook:
	jupyter lab --ip=* --port $(JUPYTER_PORT) --allow-root

run_lab:
	jupyter lab --ip=* --port $(JUPYTER_PORT) --allow-root
