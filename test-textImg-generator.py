from transformers import GPT2LMHeadModel, GPT2Tokenizer
from PIL import Image, ImageDraw, ImageFont

def generate_text_image(input_text):
    try:
        # Charger le modèle et le tokenizer
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Générer du texte avec le modèle
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        attention_mask = input_ids.clone().fill_(1)  # Créer un masque d'attention
        output = model.generate(input_ids, attention_mask=attention_mask, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95, do_sample=True, pad_token_id=tokenizer.eos_token_id)

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Créer une image avec Pillow
        image = Image.new("RGB", (500, 300), color="white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        draw.text((50, 100), generated_text, fill="black", font=font)

        # Afficher l'image
        image.show()

        # Enregistrer l'image
        image.save("output_image.png")

    except Exception as e:
        print(f"Une erreur s'est produite : {e}")

if __name__ == "__main__":
    user_input = input("Entrez le texte d'entrée : ")
    generate_text_image(user_input)
