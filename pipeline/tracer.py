import time
import os
from typing import Dict, Any, List, Optional

class TraceStep:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.start_time = time.time()
        self.end_time = None
        self.duration = None
        self.metadata = {}

    def finish(self, metadata: Dict[str, Any] = None):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        if metadata:
            self.metadata.update(metadata)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "duration_ms": round(self.duration * 1000, 2) if self.duration else 0.0,
            "metadata": self.metadata
        }

class Tracer:
    def __init__(self, session_id: Optional[str] = None):
        self.steps: List[TraceStep] = []
        self.start_time = time.time()
        self.session_id = session_id
        
        # Optional Langfuse Client integration
        self.langfuse = None
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            try:
                from langfuse import Langfuse
                self.langfuse = Langfuse()
                print("[+] Langfuse observability logger initialized.")
            except ImportError:
                print("[!] Langfuse package not installed. Skipping Langfuse initialization.")

    def start_step(self, name: str, description: str) -> TraceStep:
        step = TraceStep(name, description)
        self.steps.append(step)
        return step

    def log_to_langfuse(self, name: str, input_str: str, output_str: str, metadata: Dict[str, Any] = None):
        """
        Sends the generation trace to Langfuse if configured.
        """
        if not self.langfuse:
            return
            
        try:
            self.langfuse.generation(
                name=name,
                input=input_str,
                output=output_str,
                metadata=metadata or {},
                session_id=self.session_id
            )
        except Exception as e:
            print(f"[!] Failed to log to Langfuse: {e}")

    def to_dict(self) -> Dict[str, Any]:
        total_duration = time.time() - self.start_time
        return {
            "total_duration_ms": round(total_duration * 1000, 2),
            "steps": [s.to_dict() for s in self.steps]
        }
