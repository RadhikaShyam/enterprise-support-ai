from pathlib import Path


class DocumentService:

    SUPPORTED_EXTENSIONS = {
        ".txt",
        ".md",
    }

    def extract_text(self, file_path: str) -> str:
        path = Path(file_path)

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {path.suffix}"
            )

        return path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    def clean_text(self, text: str) -> str:
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        return "\n".join(lines)