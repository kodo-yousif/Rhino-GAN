# nose-manipulation

## Running steps

Note: your GPU must support driver 11.7 to run this docker file successfully

```terminal
docker build -t nose-ai .
```

Execute in power shell

```terminal
docker run -d --gpus all --privileged -it -p 8000:3000 -p 8001:3001 `
  -v "${PWD}:/ai" `
  --name nose-ai-container `
  nose-ai
```

```terminal
docker exec -it nose-ai-container bash

/opt/conda/bin/conda init bash

echo -e "source /opt/conda/etc/profile.d/conda.sh\nconda activate nose-ai" >> ~/.bashrc

exit

docker exec -it nose-ai-container bash
```

if conda env nose-ai is not activates run this

```terminal
conda activate nose-ai
```
