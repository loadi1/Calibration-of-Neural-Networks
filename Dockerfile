# Наследуемся от тяжёлого образа
FROM mytorch:base-v3

ENV PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
