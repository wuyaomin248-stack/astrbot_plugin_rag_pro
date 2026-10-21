# Errors

## 401

`error_code_401` means the embedding provider rejected the credentials.
The most common causes are a missing API key, an expired key, or a base URL
that points at the wrong endpoint.

To fix it:

- Open the AstrBot provider configuration and re-enter the API key.
- Confirm the base URL matches the provider that issued the key.
- Run `/rag rebuild` after the credentials are corrected; cached vectors
  computed with the old credentials are reused otherwise.

If the error persists with a valid key, the account may not have access to
the embedding model you selected. Choose a different embedding model.
