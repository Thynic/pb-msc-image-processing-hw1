# msc-image-processing-hw1

## run on collab

- import the ipynb notebook inside for-collab folder to google collab

### or

https://colab.research.google.com/drive/12GvjsU8Z0Z6zXQnUS6qeahxI3656NWpF?usp=sharing

## run on docker

### install docker compose

```bash
sudo apt install docker-compose
```

### run docker configurations

```bash
docker-compose up
```

## run on local

### install python

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

```bash
sudo apt update && sudo apt install python3
```

#### to verify

```bash
python3 --version
```

### install pip

```bash
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
```

```bash
python3 get-pip.py
```

#### or

```bash
sudo apt install python3-pip
```

#### to verify

```bash
pip --version
```

### install necessary packages

```bash
pip install -r requirements.txt
```

### apply the transformations

```bash
python3 src/main.py
```
