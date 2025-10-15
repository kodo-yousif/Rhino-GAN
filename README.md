# nose-manipulation

## Running steps

Note: your GPU must support driver 11.7 to run this docker file successfully

```terminal
docker build -t nose-ai .
```

Execute in power shell

Execute this command in the root folder of this project

```powershell
docker run -d --gpus all --privileged -it -p 8000:3000 -p 8001:3001 -v .:/nose-ai --name nose-ai-container nose-ai
```

```terminal
docker exec -it nose-ai-container bash

/opt/conda/bin/conda init bash

echo -e "source /opt/conda/etc/profile.d/conda.sh\nconda activate nose-ai" >> ~/.bashrc

exit
```

## Step 2

download this link `https://drive.google.com/file/d/1AT6bNR2ppK8f2ETL_evT27f3R_oyWNHS/view?usp=sharing` into `backend/pretrained_models`

if conda env nose-ai is not activates run this

```terminal
conda activate nose-ai
```

```terminal
cd frontend

yarn

yarn dev
```

Optional data samples or manularu download https://drive.google.com/drive/folders/15jsR9yy_pfDHiS9aE3HcYDgwtBbAneId content into backend/images/unprocessed

```terminal
python backend/images/unprocessed/populate.py
```

```terminal
conda env update -f environment.yml
conda env export > kodo.yml
fastapi run backend/main.py --host 0.0.0.0 --port 3000
```
