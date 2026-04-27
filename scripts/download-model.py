# DataMind AI - Script de telechargement du modele embarque
# Executez : python scripts/download-model.py

import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_URL = "https://huggingface.co/bartowski/google_gemma-4-4b-it-GGUF/resolve/main/google_gemma-4-4b-it-Q4_K_M.gguf"
MODEL_FILE = os.path.join(MODEL_DIR, "gemma-4-4b-it-Q4_K_M.gguf")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_FILE):
        size_mb = os.path.getsize(MODEL_FILE) / (1024 * 1024)
        print(f"Le modele existe deja ({size_mb:.0f} Mo) : {MODEL_FILE}")
        response = input("Voulez-vous le retelecharger ? (y/N) : ")
        if response.lower() != "y":
            print("Annule.")
            return

    print("Telechargement du modele Gemma 4 E4B (Q4_K_M)...")
    print(f"URL: {MODEL_URL}")
    print(f"Destination: {MODEL_FILE}")
    print("Taille approximative : ~3-4 Go")
    print("Cela peut prendre quelques minutes selon votre connexion...")
    print()

    try:
        import urllib.request

        def report(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(downloaded / total_size * 100, 100)
                downloaded_mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                print(
                    f"\r  Progression: {pct:.1f}% ({downloaded_mb:.0f}/{total_mb:.0f} Mo)",
                    end="",
                    flush=True,
                )

        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE, reporthook=report)
        print()
        print(f"[OK] Modele telecharge : {MODEL_FILE}")

        size_mb = os.path.getsize(MODEL_FILE) / (1024 * 1024)
        print(f"Taille : {size_mb:.0f} Mo")
        print()
        print("Pour utiliser le modele embarque, modifiez votre .env :")
        print("  LLM_PROVIDER=embedded")
        print("  EMBEDDED_MODEL_PATH=models/gemma-4-4b-it-Q4_K_M.gguf")

    except Exception as e:
        print(f"\n[ERREUR] Telechargement echoue : {e}")
        print()
        print("Alternatives :")
        print("  1. Telechargez manuellement depuis :")
        print(f"     {MODEL_URL}")
        print(f"  2. Placez le fichier dans : {MODEL_DIR}")
        print("  3. Ou utilisez Ollama : ollama pull gemma4:e4b")


if __name__ == "__main__":
    main()
