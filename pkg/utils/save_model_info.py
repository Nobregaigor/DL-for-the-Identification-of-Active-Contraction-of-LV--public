import json
def save_model_info(dir, features, labels, scales=[]):
  model_info = {"features": features, "labels": labels, "scales": scales}
  model_info = json.dumps(model_info)
  if not os.path.exists(dir):
    os.makedirs(dir)
  with open(dir/"info.json", "w") as jsonfile:
    jsonfile.write(model_info)