

# shared minio

```bash
# running locally
PREFIX="/opt/app-root/src/ai-mazing-race/bootstrap/"

# running remotely
#PREFIX="https://github.com/rh-aiservices-bu/ai-mazing-race
oc delete ns shared-minio
oc apply -k "${PREFIX}/shared-minio/"
```


