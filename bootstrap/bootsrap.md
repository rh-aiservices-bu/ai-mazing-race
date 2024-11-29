

# shared minio

```bash
# running locally
# PREFIX="/opt/app-root/src/ai-mazing-race/bootstrap/"
PREFIX="./bootstrap/"


# running remotely
#PREFIX="https://github.com/rh-aiservices-bu/ai-mazing-race
oc delete ns shared-minio
oc apply -k "${PREFIX}/shared-minio/"

oc delete ns user1 user2 user3 user4 user5 user-projects
oc apply -k "${PREFIX}/user-projects/"

```



```
oc -n shared-minio delete job create-buckets
oc apply -n shared-minio -f "${PREFIX}/shared-minio/create-buckets.yaml"

oc -n shared-minio logs -l job-name=create-buckets

oc -n shared-minio get route minio-console
echo "https://$(oc -n shared-minio get route minio-console  -o jsonpath='{.spec.host}')/"