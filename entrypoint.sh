#!/bin/bash
caddy reverse-proxy --from yoso.k8.devfactory.com --to localhost:5000 &
flask run --host 0.0.0.0