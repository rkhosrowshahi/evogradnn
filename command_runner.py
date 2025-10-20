#!/usr/bin/env python3
"""
Command Runner - A Python program to execute a list of commands

This script allows you to run a list of shell commands sequentially or in parallel,
with proper error handling, logging, and result reporting.
"""

import subprocess
import sys
import time
import threading
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import yaml


@dataclass
class CommandResult:
    """Data class to store command execution results"""
    command: str
    returncode: int
    stdout: str
    stderr: str
    execution_time: float
    success: bool


class CommandRunner:
    """A class to run shell commands with various execution modes"""
    
    def __init__(self, timeout: Optional[int] = None, verbose: bool = True, show_output: bool = True):
        """
        Initialize the CommandRunner
        
        Args:
            timeout: Maximum time in seconds to wait for each command (None for unlimited)
            verbose: Whether to print detailed output during execution
            show_output: Whether to show real-time command output (not captured)
        """
        self.timeout = timeout
        self.verbose = verbose
        self.show_output = show_output
        self.results: List[CommandResult] = []
    
    def run_command(self, command: str) -> CommandResult:
        """
        Execute a single command and return the result
        
        Args:
            command: The shell command to execute
            
        Returns:
            CommandResult object containing execution details
        """
        if self.verbose:
            print(f"Executing: {command}")
        
        start_time = time.time()
        
        try:
            # Run the command (with unlimited time if timeout is None)
            if self.show_output:
                # Show real-time output - don't capture stdout/stderr
                if self.timeout is None:
                    process = subprocess.run(
                        command,
                        shell=True,
                        text=True
                    )
                else:
                    process = subprocess.run(
                        command,
                        shell=True,
                        text=True,
                        timeout=self.timeout
                    )
                
                # Since we didn't capture output, set empty strings
                stdout_content = ""
                stderr_content = ""
            else:
                # Capture output silently (original behavior)
                if self.timeout is None:
                    process = subprocess.run(
                        command,
                        shell=True,
                        capture_output=True,
                        text=True
                    )
                else:
                    process = subprocess.run(
                        command,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout
                    )
                
                stdout_content = process.stdout
                stderr_content = process.stderr
            
            execution_time = time.time() - start_time
            success = process.returncode == 0
            
            result = CommandResult(
                command=command,
                returncode=process.returncode,
                stdout=stdout_content,
                stderr=stderr_content,
                execution_time=execution_time,
                success=success
            )
            
            if self.verbose:
                status = "✓" if success else "✗"
                print(f"\n{status} Command completed in {execution_time:.2f}s (exit code: {process.returncode})")
                if not self.show_output:
                    # Only show captured output if we weren't showing real-time output
                    if stdout_content:
                        print(f"  Output: {stdout_content.strip()}")
                    if stderr_content and not success:
                        print(f"  Error: {stderr_content.strip()}")
            
            return result
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            result = CommandResult(
                command=command,
                returncode=-1,
                stdout="",
                stderr=f"Command timed out after {self.timeout} seconds",
                execution_time=execution_time,
                success=False
            )
            
            if self.verbose:
                print(f"✗ Command timed out after {self.timeout}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = CommandResult(
                command=command,
                returncode=-2,
                stdout="",
                stderr=str(e),
                execution_time=execution_time,
                success=False
            )
            
            if self.verbose:
                print(f"✗ Command failed with exception: {e}")
            
            return result
    
    def run_commands_sequential(self, commands: List[str], stop_on_error: bool = False) -> List[CommandResult]:
        """
        Run commands sequentially (one after another)
        
        Args:
            commands: List of shell commands to execute
            stop_on_error: Whether to stop execution if a command fails
            
        Returns:
            List of CommandResult objects
        """
        results = []
        
        for i, command in enumerate(commands, 1):
            if self.verbose:
                print(f"\n[{i}/{len(commands)}] Running command...")
            
            result = self.run_command(command)
            results.append(result)
            
            if not result.success and stop_on_error:
                if self.verbose:
                    print(f"Stopping execution due to failed command: {command}")
                break
        
        self.results.extend(results)
        return results
    
    def run_commands_parallel(self, commands: List[str], max_workers: int = 4) -> List[CommandResult]:
        """
        Run commands in parallel using thread pool
        
        Args:
            commands: List of shell commands to execute
            max_workers: Maximum number of concurrent threads
            
        Returns:
            List of CommandResult objects
        """
        results = []
        
        if self.verbose:
            print(f"Running {len(commands)} commands in parallel (max {max_workers} workers)...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all commands
            future_to_command = {executor.submit(self.run_command, cmd): cmd for cmd in commands}
            
            # Collect results as they complete
            for future in as_completed(future_to_command):
                result = future.result()
                results.append(result)
        
        # Sort results by original command order
        command_order = {cmd: i for i, cmd in enumerate(commands)}
        results.sort(key=lambda x: command_order[x.command])
        
        self.results.extend(results)
        return results
    
    def print_summary(self, results: List[CommandResult]):
        """Print a summary of command execution results"""
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        total_time = sum(r.execution_time for r in results)
        
        print(f"\n{'='*60}")
        print("EXECUTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total commands: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total execution time: {total_time:.2f}s")
        print(f"{'='*60}")
        
        if failed > 0:
            print("\nFAILED COMMANDS:")
            for result in results:
                if not result.success:
                    print(f"  ✗ {result.command}")
                    print(f"    Exit code: {result.returncode}")
                    print(f"    Error: {result.stderr.strip()}")
    
    def save_results_to_json(self, filename: str):
        """Save execution results to a JSON file"""
        data = []
        for result in self.results:
            data.append({
                'command': result.command,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': result.execution_time,
                'success': result.success
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {filename}")


def load_commands_from_file(filename: str) -> List[str]:
    """Load commands from a text file (one command per line), JSON config, or YAML config"""
    if filename.endswith('.json'):
        return load_commands_from_json(filename)
    elif filename.endswith('.yaml') or filename.endswith('.yml'):
        return load_commands_from_yaml(filename)
    
    with open(filename, 'r') as f:
        commands = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return commands


def load_commands_from_json(filename: str) -> List[str]:
    """Load commands from a JSON configuration file"""
    with open(filename, 'r') as f:
        config = json.load(f)
    
    commands = []
    for cmd_config in config.get('commands', []):
        if isinstance(cmd_config, dict):
            commands.append(cmd_config['command'])
        else:
            commands.append(str(cmd_config))
    
    return commands


def load_commands_from_yaml(filename: str) -> List[str]:
    """Load commands from a YAML configuration file"""
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    
    commands = []
    for cmd_config in config.get('commands', []):
        if isinstance(cmd_config, dict):
            commands.append(cmd_config['command'])
        else:
            commands.append(str(cmd_config))
    
    return commands


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description="Run a list of shell commands")
    parser.add_argument('commands', nargs='*', help='Commands to execute')
    parser.add_argument('-f', '--file', help='Read commands from file (one per line)')
    parser.add_argument('-p', '--parallel', action='store_true', help='Run commands in parallel')
    parser.add_argument('-w', '--workers', type=int, default=4, help='Max parallel workers (default: 4)')
    parser.add_argument('-t', '--timeout', type=int, help='Timeout in seconds for each command')
    parser.add_argument('-s', '--stop-on-error', action='store_true', help='Stop on first error (sequential only)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Quiet mode (less verbose output)')
    parser.add_argument('-o', '--output', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Get commands from arguments or file
    if args.file:
        try:
            commands = load_commands_from_file(args.file)
        except FileNotFoundError:
            print(f"Error: File '{args.file}' not found")
            sys.exit(1)
    elif args.commands:
        commands = args.commands
    else:
        # Interactive mode - ask for commands
        print("Enter commands (one per line, empty line to finish):")
        commands = []
        while True:
            try:
                command = input("> ").strip()
                if not command:
                    break
                commands.append(command)
            except KeyboardInterrupt:
                print("\nAborted by user")
                sys.exit(1)
    
    if not commands:
        print("No commands to execute")
        sys.exit(1)
    
    # Create runner and execute commands
    runner = CommandRunner(timeout=args.timeout, verbose=not args.quiet, show_output=not args.quiet)
    
    if args.parallel:
        results = runner.run_commands_parallel(commands, max_workers=args.workers)
    else:
        results = runner.run_commands_sequential(commands, stop_on_error=args.stop_on_error)
    
    # Print summary
    runner.print_summary(results)
    
    # Save results if requested
    if args.output:
        runner.save_results_to_json(args.output)
    
    # Exit with error code if any command failed
    failed_count = sum(1 for r in results if not r.success)
    sys.exit(min(failed_count, 1))


if __name__ == "__main__":
    # Example usage when run directly
    if len(sys.argv) == 1:
        print("Command Runner - Example Usage")
        print("=" * 40)
        
        # Example commands
        example_commands = [
            "echo 'Hello, World!'",
            "python --version",
            "ls -la" if sys.platform != "win32" else "dir",
            "echo 'Task completed'"
        ]
        
        print("Running example commands...")
        runner = CommandRunner(verbose=True, show_output=True)
        results = runner.run_commands_sequential(example_commands)
        runner.print_summary(results)
        
        print("\nTo use this script:")
        print("python command_runner.py 'echo hello' 'python --version'")
        print("python command_runner.py -f commands.txt")
        print("python command_runner.py --parallel 'echo 1' 'echo 2' 'echo 3'")
    else:
        main()
