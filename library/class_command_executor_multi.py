import psutil
import subprocess
from library.class_command_executor import CommandExecutor, log
from library.port_util import mark_port_as_used, remove_port_from_used


class MultiCommandExecutor(CommandExecutor):
    """
    Command executor that manages multiple processes
    """
    
    def __init__(self):
        super().__init__()
        self.processes = {}
        self.main_process_port = 0
        
    def execute_command(self, run_cmd, port:int=0):
        if port != 0:
            mark_port_as_used(port)
            self.main_process_port = port
        
        return super().execute_command(run_cmd)
    
    def kill_all_commands(self):
        for ports, proc in self.processes.items():
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
                    remove_port_from_used(ports)
            else:
                log.info('There is no running process to kill.')
        self.processes.clear()
    
    def kill_command(self, port:int=-1):
        if port != 'all' and not isinstance(port, int):
            try:
                port = int(port)
            except Exception as e:
                log.info(f'Invalid port: {port}')
                return
        if port == -1 or port == 'all' or port is None:
            self.kill_all_commands()
            return super().kill_command()
        else:
            return self.kill_port_command(port)
    
    def execute_command_subprocess(self, run_cmd, port:int) -> int:
        """
        Runs a command in a subprocess.
        Returns the port that the command is running on.
        """
        log.info(f'Running command: {run_cmd} on port {port}')
        assert port not in self.processes, f'Port {port} is already in use.'
        proc = subprocess.Popen(run_cmd, shell=True)
        self.processes[port] = proc
        mark_port_as_used(port)
        return port
    
    def kill_port_command(self, port:int):
        """
        Kills a command running on a port.
        """
        if port in self.processes:
            proc = self.processes[port]
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
            self.processes.pop(port)
        else:
            log.info(f'Port {port} is not in use.')
            
    def get_running_process_ports(self):
        return list(self.processes.keys())