import subprocess
import sys
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class RestartHandler(FileSystemEventHandler):
    def __init__(self, script):
        self.script = script
        self.process = None
        self.start_server()

    def start_server(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
        print('\n🔄 Starting server...\n')
        self.process = subprocess.Popen([sys.executable, self.script])

    def on_modified(self, event):
        if event.src_path.endswith('.py'):
            print(f'\n📝 Detected change in {event.src_path}')
            self.start_server()

if __name__ == '__main__':
    handler = RestartHandler('wine_reviews.py')
    observer = Observer()
    observer.schedule(handler, '.', recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        if handler.process:
            handler.process.terminate()
    observer.join()
