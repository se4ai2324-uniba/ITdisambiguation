import requests
from pathlib import Path
import shutil


def delete_directory(path):
    """Elimina la directory specificata e tutto il suo contenuto."""
    try:
        shutil.rmtree(path)
        print(f"La directory '{path}' è stata eliminata.")
    except OSError as e:
        print(f"Errore: {e.strerror}")


def download_image(image_url):
    """Scarica un'immagine dall'URL fornito e la salva in locale."""
    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        file_path = Path("downloaded_images") / Path(image_url).name
        with open(file_path, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        return file_path
    return None

def send_context_to_api(img_path, target_word, contexts):
    api_url = "http://127.0.0.1:8000/models/RN50/predict_context"
    #img = open("data/Test/resized_test_images_N/image.4457.jpg", "rb")
    img = open(img_path, "rb")

    response = requests.post(
        api_url,
        files={"image": img},
        data={"target_word": target_word, "contexts": contexts}
    )
    img.close()
    return response

def send_images_to_api(api_url, image_urls, target_word, context):
    """Scarica le immagini e le invia all'API."""
    # Assicurati che la cartella delle immagini scaricate esista
    Path("downloaded_images").mkdir(exist_ok=True)

    # Scarica le immagini
    files = []
    for url in image_urls:
        image_path = download_image(url)
        if image_path:
            files.append(('images', (image_path.name, open(image_path, 'rb'), 'image/jpeg')))

    # Invia le immagini all'API
    response = requests.post(
        api_url,
        files=files,
        data={'target_word': target_word, 'context': context}
    )

    # Chiudi i file
    for _, file_tuple in files:
        file_tuple[1].close()
    
    directory_path = Path("downloaded_images")
    delete_directory(directory_path)

    return response
