First start by installing everything in the requirements.lock file
Then install ray with the following command
    pip install "ray[default]"

Download prometheus and Grafana for the dashboards to work

https://docs.ray.io/en/latest/ray-observability/ray-metrics.html


Prometheus needs to be just downloaded, unzipped and run using these commands. latest version can be found at https://prometheus.io/download/
    wget https://github.com/prometheus/prometheus/releases/download/v2.43.0/prometheus-2.43.0.linux-amd64.tar.gz
    tar xvfz prometheus-*.tar.gz
    cd prometheus-*
    ./prometheus --config.file=/tmp/ray/session_latest/metrics/prometheus/prometheus.yml

get grafana binary and unzip it.
    wget https://dl.grafana.com/enterprise/release/grafana-enterprise-9.4.7.linux-amd64.tar.gz
    tar -zxvf grafana-enterprise-9.4.7.linux-amd64.tar.gz

Run grafana using below. grafana server will be running on port 3000. an alternate port can be passed using --http-port 3001 for example
    cd grafana-9.4.7/
    ./bin/grafana-server --config /tmp/ray/session_latest/metrics/grafana/grafana.ini web

Forward ports 3000 and 8265 to your local machine. easiest with VScode. Ray dashboard should be available on localhost:8265. This will not be available until you actually start the training.


of course have to start the program with the following command as in the README.md
    python muzero.py