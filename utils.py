from cryptography.fernet import Fernet
import os


def decrypt_huggingface_token(decryption_key_environment_variable_name="FERNET_DECRYPTION_KEY", file_name="huggingface_key.bin"):
    # 1. Retrieve the encryption key (from an environment variable)

    try:
        encryption_key = os.environ.get(decryption_key_environment_variable_name)
        if encryption_key is None:
            raise ValueError(f"Environment variable {decryption_key_environment_variable_name} not set.")
        encryption_key = encryption_key.encode() # Ensure it's bytes
    except Exception as e:
        print(f"Error: {e}")
        print(f"Please set the {decryption_key_environment_variable_name} environment variable. Example: export {decryption_key_environment_variable_name}=\"your_generated_key\"")
        # As a fallback (less secure for production), you could prompt:
        # encryption_key = input("Enter your decryption key: ").encode()
        # If you go this route, consider using `getpass` to hide input:
        # import getpass
        # encryption_key = getpass.getpass("Enter your decryption key: ").encode()
        exit("Cannot proceed without decryption key.") # Exit if key is missing

    # 2. Initialize the Fernet cipher suite
    cipher_suite = Fernet(encryption_key)

    # 3. Load the encrypted API key from the file
    try:
        with open(file_name, 'rb') as f:
            encrypted_hf_token_data = f.read()
    except FileNotFoundError:
        print(f"Error: {file_name} not found. Please ensure the encrypted file exists.")
        exit("Cannot proceed without encrypted token file.")

    # 4. Decrypt the API key
    try:
        decrypted_hf_token = cipher_suite.decrypt(encrypted_hf_token_data).decode()
        print("Hugging Face token decrypted successfully.")
    except Exception as e:
        print(f"Error decrypting token: {e}")
        print("This might be due to an incorrect decryption key or corrupted file.")
        exit("Decryption failed.")

    # 5. Use the decrypted token
    return decrypted_hf_token