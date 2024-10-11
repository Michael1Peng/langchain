import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import csv

# Load environment variables
load_dotenv()

# Database connection settings
DB_USER = os.getenv("DB_USER", "wsl_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "a124567")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "media_crawler")

# Create SQLAlchemy engine
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")

# Create a base class for declarative models
Base = declarative_base()


# Define the XhsNote model (same as in plot_covid.py)
class XhsNote(Base):
    __tablename__ = "xhs_note"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String)
    nickname = Column(String)
    avatar = Column(String)
    ip_location = Column(String)
    add_ts = Column(BigInteger)
    last_modify_ts = Column(BigInteger)
    note_id = Column(String)
    type = Column(String)
    title = Column(String)
    desc = Column(Text)
    video_url = Column(String)
    time = Column(BigInteger)
    last_update_time = Column(BigInteger)
    liked_count = Column(String)
    collected_count = Column(String)
    comment_count = Column(String)
    share_count = Column(String)
    image_list = Column(Text)
    tag_list = Column(Text)
    note_url = Column(String)
    is_processed_llm = Column(Integer, default=0)
    is_about_covid = Column(Integer, default=0)
    is_about_fever = Column(Integer, default=0)
    is_about_virus = Column(Integer, default=0)
    is_about_sick = Column(Integer, default=0)
    is_sick_recent = Column(Integer, default=0)


# Create a session factory
Session = sessionmaker(bind=engine)


def export_xhs_notes_to_csv(output_dir="custom-examples/xhs-notes"):
    session = Session()
    try:
        # 选择 desc 中带有“旅游”的条目，按时间降序排列，并只取前十条
        notes = (
            session.query(XhsNote)
            .filter(XhsNote.desc.contains("旅游"))
            .order_by(XhsNote.time.desc())
            .limit(10)
            .all()
        )

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 将每条记录写入单独的文件
        for note in notes:
            # Trim and sanitize the title for use as a filename
            safe_title = "".join(
                c for c in note.title if c.isalnum() or c in (" ", "_", "-")
            )
            safe_title = safe_title.strip()[:50]  # Limit to 50 characters
            file_path = os.path.join(output_dir, f"{safe_title}.txt")
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(note.desc)

        print(f"Export completed. Files saved to {output_dir}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        session.close()


if __name__ == "__main__":
    export_xhs_notes_to_csv()
