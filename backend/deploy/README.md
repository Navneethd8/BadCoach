# Deploy sync (inference category)

These files are **overwritten** whenever you run:

```bash
cd backend
python3 -m api.inference_model_cli set <category>
```


| File                    | Purpose                                                                                                                        |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `docker-inference.env`  | `ISOCOURT_INFERENCE_CATEGORY` for `**docker compose**` (see `../docker-compose.yml`).                                          |
| `ci_inference_category` | Single line: category id only. GitHub Actions reads it into `GITHUB_ENV` before copying the single checkpoint to Hugging Face. |


Commit the updates with `inference_selection.json` so local Docker, CI, and the Space stay aligned.