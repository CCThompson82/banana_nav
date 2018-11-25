FROM ccthompson82/drlnd:0.0.5

################################################################################
#  Install Headless Unity Banana Navigation Environment
################################################################################
WORKDIR /tmp
RUN wget -q https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip && \
    unzip Banana_Linux_NoVis.zip -d /usr/lib/

ENV UNITY_ENV_PATH="/usr/lib/Banana_Linux_NoVis/Banana.x86_64"

################################################################################
#  setup environment
################################################################################
USER $UID
WORKDIR /workdir
ENV ROOT_DIR=/workdir
ENTRYPOINT ["/bin/bash"]
