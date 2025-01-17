from dotenv import load_dotenv


def init():
    load_dotenv(".env", override=True)
    # print("OPENAI_API_KEY", os.environ["OPENAI_API_KEY"])
