FROM grafana/grafana:9.5.2
COPY config/grafana/datasources.yml /etc/grafana/provisioning/datasources/datasources.yml
COPY config/grafana/dashboards.yml /etc/grafana/provisioning/dashboards/dashboards.yml
COPY config/grafana/dashboard_summarization.json /var/lib/grafana/dashboards/dashboard_summarization.json
