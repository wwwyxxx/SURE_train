FROM docker.v2.aispeech.com/sjtu/sjtu_yukai-yixuanwang-kimiaudio:v0

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# TASU-specific runtime dependencies (SenseVoice frontend + audio I/O).
RUN pip install --no-cache-dir \
    funasr \
    kaldiio \
    soundfile \
    sentencepiece \
    scipy \
    pandas \
    h5py \
    einops

WORKDIR /workspace
CMD ["/bin/bash"]
