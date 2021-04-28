# This Dockerfile has been adapted from the VS Code example at https://code.visualstudio.com/docs/containers/quickstart-python

# Base image is a lightweight version of Python
FROM continuumio/miniconda3

# Expose the port on which our server will run
EXPOSE 5000

# Keeps Python from generating .pyc files in the container
#ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-c"]

# Set the default working directory, then copy the Python source code into it
WORKDIR /app
COPY . /app

# cron setup
#------------------------------------
# Get cron
RUN apt-get update && apt-get -y install cron
# Setup cron job
ADD ./cronconfig /etc/cron.d/cronconfig
# Give execution rights on the cron job
RUN chmod 0644 /etc/cron.d/cronconfig
# Give execution rights to the script
RUN chmod 0744 /app/delete_logs.sh
# Apply cron job
RUN crontab /etc/cron.d/cronconfig
# Create the log file to be able to run tail
RUN touch /var/log/cron.log
#------------------------------------

# conda setup
#------------------------------------
ARG conda_env=oscml_venv
# Install the required Python libraries
RUN ./install_script.sh -v -n $conda_env -i -e
# Make RUN commands use the new environment:
RUN echo "source activate $conda_env" > ~/.bashrc
ENV PATH /opt/conda/envs/$conda_env/bin:$PATH
RUN conda install openjdk
#------------------------------------

# Switch to a non-root user before running the server, for security reasons
# (See https://code.visualstudio.com/docs/containers/python-user-rights)
#RUN useradd appuser && chown -R appuser /app
#USER appuser
# Set the entrypoint
RUN chmod 0744 /app/app_entry_point.sh
ENTRYPOINT /app/app_entry_point.sh