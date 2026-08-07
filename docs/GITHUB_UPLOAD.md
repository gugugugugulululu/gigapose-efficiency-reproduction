# Publishing the repository

```bash
cd gigapose-efficiency-reproduction
git init
git add .
git commit -m "Initial reproducible LM-O Top-3 and adaptive release"
git branch -M main
git remote add origin <your-empty-github-repository-url>
git push -u origin main
```

Before pushing:

1. Replace `<your-repository-url>` in `README.md`.
2. Check that no checkpoint, dataset, Google Drive credential, private token or result archive is present.
3. Run `make test` and `make check`.
4. Add the final GitHub URL to the dissertation.
5. Create a release tag such as `v0.1.0` so the submitted report points to a frozen version.

Suggested repository description:

> Reproduction code for LM-O GigaPose–MegaPose Top-3 and object-adaptive inference-efficiency experiments.
