#!/bin/bash

gunicorn -w 4 'app:create_app()' -b 0.0.0.0 
