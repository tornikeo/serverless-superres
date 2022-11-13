
import banana_dev as banana, os, dotenv
model_inputs = dict(locals())

dotenv.load_dotenv()

api_key = os.getenv("BANANA_API_KEY")
model_key = os.getenv("BANANA_SUPERRES_MODEL_KEY")

out = banana.run(api_key, model_key, {'image':"https://i.imgur.com/IdcBNz7.jpg"})

print(out)