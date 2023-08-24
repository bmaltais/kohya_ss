import subprocess
import psutil
from library.custom_logging import setup_logging
from library.port_util import mark_port_as_used, remove_port_from_used

# Set up logging
log = setup_logging()


class CommandExecutor:
    def __init__(self):
        self.process = None
        self.port_dict = {} # {port: process}
        self.main_process_port = 0

    def execute_command(self, run_cmd, port:int=0):
        if self.process and self.process.poll() is None:
            log.info(
                'The command is already running. Please wait for it to finish.'
            )
        else:
            self.process = subprocess.Popen(run_cmd, shell=True)
        if port != 0:
            mark_port_as_used(port)
            self.main_process_port = port

    def kill_command(self):
        if self.process and self.process.poll() is None:
            try:
                parent = psutil.Process(self.process.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
                log.info('The running process has been terminated.')
            except psutil.NoSuchProcess:
                log.info('The process does not exist.')
            except Exception as e:
                log.info(f'Error when terminating process: {e}')
        else:
            log.info('There is no running process to kill.')
        if self.main_process_port != 0:
            remove_port_from_used(self.main_process_port)
            self.main_process_port = 0
        self.kill_all_commands()
            
    def execute_command_subprocess(self, run_cmd, port:int):
        """
        Runs a command in a subprocess.
        """
        log.info(f'Running command: {run_cmd} on port {port}')
        assert port not in self.port_dict, f'Port {port} is already in use.'
        proc = subprocess.Popen(run_cmd, shell=True)
        self.port_dict[port] = proc
        mark_port_as_used(port)
        return proc
    
    def kill_all_commands(self):
        """
        Kills all running commands.
        """
        for port in self.port_dict:
            proc = self.port_dict[port]
            if proc and proc.poll() is None:
                try:
                    parent = psutil.Process(proc.pid)
                    for child in parent.children(recursive=True):
                        child.kill()
                    parent.kill()
                    log.info('The running process has been terminated.')
                except psutil.NoSuchProcess:
                    log.info('The process does not exist.')
                except Exception as e:
                    log.info(f'Error when terminating process: {e}')
                finally:
                    remove_port_from_used(port)
            else:
                log.info('There is no running process to kill.')
            del self.port_dict[port]
            
