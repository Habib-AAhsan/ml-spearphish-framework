# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose Flask port
EXPOSE 10000

# Run the app
CMD ["gunicorn", "app_ui:app", "--bind", "0.0.0.0:10000"]

