#!/usr/bin/env python3
"""
Advanced code editor component for the RL_CLI system.
Provides a text-based interface for editing code with syntax highlighting,
auto-completion, and integration with the agent's self-modification capabilities.
"""

import curses
import os
import sys
import re
import asyncio
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Tuple
import pygments
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
from pygments.token import Token

class EditorMode(Enum):
    """Editor operation modes"""
    NORMAL = auto()  # Normal editing mode
    COMMAND = auto() # Command entry mode
    SEARCH = auto()  # Search mode
    HELP = auto()    # Help display mode

class Editor:
    """
    Text-based code editor with syntax highlighting and agent integration.
    Features:
    - Syntax highlighting for Python code
    - Line numbers
    - Search functionality
    - Integration with agent's code analysis
    - Auto-completion suggestions
    - Command mode for editor operations
    """
    def __init__(self, file_path: str, agent=None):
        self.file_path = file_path
        self.agent = agent
        self.lines = []
        self.cursor_y = 0
        self.cursor_x = 0
        self.scroll_y = 0
        self.mode = EditorMode.NORMAL
        self.status_message = ""
        self.command_buffer = ""
        self.search_buffer = ""
        self.search_results = []
        self.current_search_idx = 0
        self.undo_stack = []
        self.redo_stack = []
        self.modified = False
        self.syntax_highlighting = True
        self.line_numbers = True
        self.tab_width = 4
        self.auto_indent = True
        self.suggestions = []
        self.showing_suggestions = False
        self.selected_suggestion = 0
        self.lexer = PythonLexer()
        self.formatter = TerminalFormatter()
        self.help_text = self._get_help_text()
        self.load_file()
        
    def load_file(self):
        """Load file content or create a new file"""
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r') as f:
                    self.lines = f.read().splitlines()
                if not self.lines:
                    self.lines = [""]
            else:
                self.lines = [""]
                self.status_message = f"New file: {self.file_path}"
        except Exception as e:
            self.status_message = f"Error loading file: {e}"
            self.lines = [""]
            
    def save_file(self):
        """Save current content to file"""
        try:
            with open(self.file_path, 'w') as f:
                f.write('\n'.join(self.lines))
            self.modified = False
            self.status_message = f"Saved: {self.file_path}"
            return True
        except Exception as e:
            self.status_message = f"Error saving file: {e}"
            return False
            
    def run(self, stdscr):
        """Main editor loop"""
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_BLUE, -1)
        curses.init_pair(4, curses.COLOR_RED, -1)
        curses.init_pair(5, curses.COLOR_MAGENTA, -1)
        curses.init_pair(6, curses.COLOR_CYAN, -1)
        curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_BLUE)  # Selected suggestion
        
        # Hide cursor
        curses.curs_set(1)
        
        # Enable keypad for special keys
        stdscr.keypad(True)
        
        running = True
        while running:
            # Clear screen
            stdscr.clear()
            
            # Get terminal dimensions
            height, width = stdscr.getmaxyx()
            
            # Draw editor content
            self._draw_editor(stdscr, height, width)
            
            # Draw status line
            self._draw_status_line(stdscr, height, width)
            
            # Draw command line if in command mode
            if self.mode == EditorMode.COMMAND:
                self._draw_command_line(stdscr, height, width)
            elif self.mode == EditorMode.SEARCH:
                self._draw_search_line(stdscr, height, width)
            elif self.mode == EditorMode.HELP:
                self._draw_help(stdscr, height, width)
                
            # Draw suggestions if showing
            if self.showing_suggestions:
                self._draw_suggestions(stdscr, height, width)
                
            # Position cursor
            if self.mode == EditorMode.NORMAL:
                # Calculate cursor position with line numbers
                line_num_width = len(str(len(self.lines))) + 1 if self.line_numbers else 0
                cursor_y_pos = self.cursor_y - self.scroll_y
                if 0 <= cursor_y_pos < height - 1:  # -1 for status line
                    stdscr.move(cursor_y_pos, self.cursor_x + line_num_width)
            elif self.mode == EditorMode.COMMAND:
                # Position cursor in command line
                stdscr.move(height - 1, len(":") + len(self.command_buffer))
            elif self.mode == EditorMode.SEARCH:
                # Position cursor in search line
                stdscr.move(height - 1, len("/") + len(self.search_buffer))
                
            # Refresh screen
            stdscr.refresh()
            
            # Handle input
            try:
                key = stdscr.getch()
                if key != -1:
                    running = self._handle_key(key, height, width)
            except curses.error:
                pass
                
            # Small sleep to reduce CPU usage
            curses.napms(10)
            
        # Save on exit if modified
        if self.modified:
            save_prompt = "Save changes before exit? (y/n): "
            stdscr.addstr(height - 1, 0, save_prompt)
            stdscr.refresh()
            
            while True:
                key = stdscr.getch()
                if key in (ord('y'), ord('Y')):
                    self.save_file()
                    break
                elif key in (ord('n'), ord('N')):
                    break
                    
        # Restore cursor visibility
        curses.curs_set(1)
            
    def _draw_editor(self, stdscr, height, width):
        """Draw the editor content with syntax highlighting"""
        # Calculate visible lines
        visible_lines = min(height - 1, len(self.lines) - self.scroll_y)
        
        # Calculate line number width
        line_num_width = len(str(len(self.lines))) + 1 if self.line_numbers else 0
        
        # Draw each visible line
        for i in range(visible_lines):
            line_idx = i + self.scroll_y
            if line_idx >= len(self.lines):
                break
                
            # Draw line number if enabled
            if self.line_numbers:
                line_num = str(line_idx + 1).rjust(line_num_width - 1) + " "
                stdscr.addstr(i, 0, line_num, curses.color_pair(3))
                
            # Get the line content
            line = self.lines[line_idx]
            
            # Apply syntax highlighting if enabled
            if self.syntax_highlighting:
                # Use pygments for syntax highlighting
                highlighted = pygments.highlight(line, self.lexer, self.formatter)
                
                # Convert ANSI escape codes to curses colors
                pos = line_num_width
                for token, text in pygments.lex(line, self.lexer):
                    color = self._token_to_color(token)
                    stdscr.addstr(i, pos, text, color)
                    pos += len(text)
            else:
                # Draw without highlighting
                stdscr.addstr(i, line_num_width, line[:width - line_num_width])
                
            # Highlight search results
            if self.search_results:
                for result_line, result_col in self.search_results:
                    if result_line == line_idx:
                        search_len = len(self.search_buffer)
                        is_current = (self.current_search_idx < len(self.search_results) and 
                                     self.search_results[self.current_search_idx] == (result_line, result_col))
                        
                        # Use different color for current search result
                        color = curses.A_REVERSE if is_current else curses.color_pair(2)
                        
                        # Highlight the search result
                        for j in range(search_len):
                            if result_col + j < width - line_num_width:
                                try:
                                    stdscr.chgat(i, line_num_width + result_col + j, 1, color)
                                except curses.error:
                                    pass
                
    def _token_to_color(self, token):
        """Convert pygments token to curses color"""
        if token in Token.Keyword:
            return curses.color_pair(1)  # Green
        elif token in Token.String:
            return curses.color_pair(2)  # Yellow
        elif token in Token.Name.Function or token in Token.Name.Class:
            return curses.color_pair(6)  # Cyan
        elif token in Token.Name.Builtin:
            return curses.color_pair(5)  # Magenta
        elif token in Token.Comment:
            return curses.color_pair(3)  # Blue
        elif token in Token.Error:
            return curses.color_pair(4)  # Red
        else:
            return curses.A_NORMAL
            
    def _draw_status_line(self, stdscr, height, width):
        """Draw the status line at the bottom of the editor"""
        status_left = f" {self.file_path} "
        if self.modified:
            status_left += "[+] "
            
        mode_str = f" {self.mode.name} "
        cursor_pos = f" Ln {self.cursor_y + 1}, Col {self.cursor_x + 1} "
        
        # Calculate available space
        available = width - len(status_left) - len(mode_str) - len(cursor_pos)
        
        # Truncate status message if needed
        status_msg = self.status_message
        if len(status_msg) > available:
            status_msg = status_msg[:available - 3] + "..."
            
        # Pad status message to fill available space
        status_msg = status_msg.ljust(available)
        
        # Draw status components
        stdscr.attron(curses.A_REVERSE)
        stdscr.addstr(height - 2, 0, status_left)
        stdscr.addstr(height - 2, len(status_left), status_msg)
        stdscr.addstr(height - 2, len(status_left) + len(status_msg), mode_str)
        stdscr.addstr(height - 2, len(status_left) + len(status_msg) + len(mode_str), cursor_pos)
        stdscr.attroff(curses.A_REVERSE)
        
    def _draw_command_line(self, stdscr, height, width):
        """Draw the command line at the bottom of the editor"""
        stdscr.addstr(height - 1, 0, ":" + self.command_buffer)
        
    def _draw_search_line(self, stdscr, height, width):
        """Draw the search line at the bottom of the editor"""
        search_status = ""
        if self.search_results:
            search_status = f" ({self.current_search_idx + 1}/{len(self.search_results)})"
            
        stdscr.addstr(height - 1, 0, "/" + self.search_buffer + search_status)
        
    def _draw_help(self, stdscr, height, width):
        """Draw the help screen"""
        # Create a centered box for help text
        help_height = min(len(self.help_text) + 4, height - 4)
        help_width = min(max(len(line) for line in self.help_text) + 4, width - 4)
        
        help_y = (height - help_height) // 2
        help_x = (width - help_width) // 2
        
        # Draw box
        for i in range(help_height):
            if i == 0 or i == help_height - 1:
                # Top and bottom borders
                stdscr.addstr(help_y + i, help_x, "+" + "-" * (help_width - 2) + "+")
            else:
                # Side borders
                stdscr.addstr(help_y + i, help_x, "|")
                stdscr.addstr(help_y + i, help_x + help_width - 1, "|")
                
        # Draw title
        title = " Help - Press ESC to close "
        stdscr.addstr(help_y, help_x + (help_width - len(title)) // 2, title)
        
        # Draw help text
        for i, line in enumerate(self.help_text[:help_height - 4]):
            stdscr.addstr(help_y + i + 2, help_x + 2, line[:help_width - 4])
            
    def _draw_suggestions(self, stdscr, height, width):
        """Draw auto-completion suggestions"""
        if not self.suggestions:
            self.showing_suggestions = False
            return
            
        # Calculate position for suggestions box
        line_num_width = len(str(len(self.lines))) + 1 if self.line_numbers else 0
        sugg_y = self.cursor_y - self.scroll_y + 1
        sugg_x = self.cursor_x + line_num_width
        
        # Ensure suggestions are visible
        if sugg_y >= height - 3:
            sugg_y = self.cursor_y - self.scroll_y - len(self.suggestions) - 1
            
        # Calculate box dimensions
        max_sugg_len = min(max(len(s) for s in self.suggestions) + 2, width - sugg_x - 2)
        sugg_height = min(len(self.suggestions) + 2, height - sugg_y - 3)
        
        # Draw suggestions box
        for i in range(sugg_height):
            if i == 0 or i == sugg_height - 1:
                # Top and bottom borders
                stdscr.addstr(sugg_y + i, sugg_x, "+" + "-" * (max_sugg_len - 2) + "+")
            else:
                # Side borders
                stdscr.addstr(sugg_y + i, sugg_x, "|")
                stdscr.addstr(sugg_y + i, sugg_x + max_sugg_len - 1, "|")
                
                # Draw suggestion if available
                sugg_idx = i - 1
                if sugg_idx < len(self.suggestions):
                    sugg = self.suggestions[sugg_idx]
                    
                    # Highlight selected suggestion
                    if sugg_idx == self.selected_suggestion:
                        stdscr.attron(curses.color_pair(7))
                        stdscr.addstr(sugg_y + i, sugg_x + 1, sugg.ljust(max_sugg_len - 2))
                        stdscr.attroff(curses.color_pair(7))
                    else:
                        stdscr.addstr(sugg_y + i, sugg_x + 1, sugg.ljust(max_sugg_len - 2))
                        
    def _handle_key(self, key, height, width):
        """Handle keyboard input based on current mode"""
        if self.mode == EditorMode.HELP:
            # In help mode, ESC returns to normal mode
            if key == 27:  # ESC
                self.mode = EditorMode.NORMAL
            return True
            
        elif self.mode == EditorMode.NORMAL:
            # Handle normal mode keys
            if key == 27:  # ESC
                # Clear status message and hide suggestions
                self.status_message = ""
                self.showing_suggestions = False
                return True
                
            elif key == ord(':'):
                # Enter command mode
                self.mode = EditorMode.COMMAND
                self.command_buffer = ""
                return True
                
            elif key == ord('/'):
                # Enter search mode
                self.mode = EditorMode.SEARCH
                self.search_buffer = ""
                self.search_results = []
                self.current_search_idx = 0
                return True
                
            elif key == curses.KEY_F1:
                # Show help
                self.mode = EditorMode.HELP
                return True
                
            elif key == curses.KEY_UP:
                # Move cursor up
                if self.showing_suggestions:
                    # Navigate suggestions
                    self.selected_suggestion = max(0, self.selected_suggestion - 1)
                else:
                    # Move cursor up
                    if self.cursor_y > 0:
                        self.cursor_y -= 1
                        # Adjust cursor x if line is shorter
                        if self.cursor_x >= len(self.lines[self.cursor_y]):
                            self.cursor_x = len(self.lines[self.cursor_y])
                        # Scroll if needed
                        if self.cursor_y < self.scroll_y:
                            self.scroll_y = self.cursor_y
                return True
                
            elif key == curses.KEY_DOWN:
                # Move cursor down
                if self.showing_suggestions:
                    # Navigate suggestions
                    self.selected_suggestion = min(len(self.suggestions) - 1, self.selected_suggestion + 1)
                else:
                    # Move cursor down
                    if self.cursor_y < len(self.lines) - 1:
                        self.cursor_y += 1
                        # Adjust cursor x if line is shorter
                        if self.cursor_x >= len(self.lines[self.cursor_y]):
                            self.cursor_x = len(self.lines[self.cursor_y])
                        # Scroll if needed
                        if self.cursor_y >= self.scroll_y + height - 2:  # -2 for status and command lines
                            self.scroll_y = self.cursor_y - height + 3
                return True
                
            elif key == curses.KEY_LEFT:
                # Move cursor left
                if self.cursor_x > 0:
                    self.cursor_x -= 1
                elif self.cursor_y > 0:
                    # Move to end of previous line
                    self.cursor_y -= 1
                    self.cursor_x = len(self.lines[self.cursor_y])
                    # Scroll if needed
                    if self.cursor_y < self.scroll_y:
                        self.scroll_y = self.cursor_y
                return True
                
            elif key == curses.KEY_RIGHT:
                # Move cursor right
                if self.cursor_x < len(self.lines[self.cursor_y]):
                    self.cursor_x += 1
                elif self.cursor_y < len(self.lines) - 1:
                    # Move to start of next line
                    self.cursor_y += 1
                    self.cursor_x = 0
                    # Scroll if needed
                    if self.cursor_y >= self.scroll_y + height - 2:
                        self.scroll_y = self.cursor_y - height + 3
                return True
                
            elif key == curses.KEY_HOME:
                # Move to start of line
                self.cursor_x = 0
                return True
                
            elif key == curses.KEY_END:
                # Move to end of line
                self.cursor_x = len(self.lines[self.cursor_y])
                return True
                
            elif key == curses.KEY_PPAGE:  # Page Up
                # Move up one page
                self.cursor_y = max(0, self.cursor_y - (height - 3))
                self.scroll_y = max(0, self.scroll_y - (height - 3))
                # Adjust cursor x if line is shorter
                if self.cursor_x >= len(self.lines[self.cursor_y]):
                    self.cursor_x = len(self.lines[self.cursor_y])
                return True
                
            elif key == curses.KEY_NPAGE:  # Page Down
                # Move down one page
                self.cursor_y = min(len(self.lines) - 1, self.cursor_y + (height - 3))
                self.scroll_y = min(len(self.lines) - (height - 2), self.scroll_y + (height - 3))
                # Adjust cursor x if line is shorter
                if self.cursor_x >= len(self.lines[self.cursor_y]):
                    self.cursor_x = len(self.lines[self.cursor_y])
                return True
                
            elif key == 10:  # Enter
                # If suggestions are showing, accept the selected one
                if self.showing_suggestions and self.suggestions:
                    self._accept_suggestion()
                    return True
                    
                # Save current state for undo
                self._save_undo_state()
                
                # Split line at cursor
                current_line = self.lines[self.cursor_y]
                self.lines[self.cursor_y] = current_line[:self.cursor_x]
                
                # Auto-indent new line if enabled
                if self.auto_indent:
                    # Calculate indentation of current line
                    indent = re.match(r'^\s*', current_line).group(0)
                    
                    # Add extra indent after certain keywords
                    if re.search(r':\s*$', current_line):
                        indent += ' ' * self.tab_width
                        
                    # Insert new line with indentation
                    self.lines.insert(self.cursor_y + 1, indent + current_line[self.cursor_x:])
                    self.cursor_y += 1
                    self.cursor_x = len(indent)
                else:
                    # Insert new line without indentation
                    self.lines.insert(self.cursor_y + 1, current_line[self.cursor_x:])
                    self.cursor_y += 1
                    self.cursor_x = 0
                    
                # Scroll if needed
                if self.cursor_y >= self.scroll_y + height - 2:
                    self.scroll_y = self.cursor_y - height + 3
                    
                self.modified = True
                return True
                
            elif key == 9:  # Tab
                # If suggestions are showing, accept the selected one
                if self.showing_suggestions and self.suggestions:
                    self._accept_suggestion()
                    return True
                    
                # Save current state for undo
                self._save_undo_state()
                
                # Insert tab (spaces)
                current_line = self.lines[self.cursor_y]
                self.lines[self.cursor_y] = current_line[:self.cursor_x] + (' ' * self.tab_width) + current_line[self.cursor_x:]
                self.cursor_x += self.tab_width
                self.modified = True
                return True
                
            elif key == 127 or key == curses.KEY_BACKSPACE:  # Backspace
                # Save current state for undo
                self._save_undo_state()
                
                if self.cursor_x > 0:
                    # Delete character before cursor
                    current_line = self.lines[self.cursor_y]
                    self.lines[self.cursor_y] = current_line[:self.cursor_x - 1] + current_line[self.cursor_x:]
                    self.cursor_x -= 1
                    self.modified = True
                elif self.cursor_y > 0:
                    # Join with previous line
                    current_line = self.lines[self.cursor_y]
                    prev_line = self.lines[self.cursor_y - 1]
                    self.cursor_x = len(prev_line)
                    self.lines[self.cursor_y - 1] = prev_line + current_line
                    self.lines.pop(self.cursor_y)
                    self.cursor_y -= 1
                    self.modified = True
                    
                    # Scroll if needed
                    if self.cursor_y < self.scroll_y:
                        self.scroll_y = self.cursor_y
                return True
                
            elif key == curses.KEY_DC:  # Delete
                # Save current state for undo
                self._save_undo_state()
                
                current_line = self.lines[self.cursor_y]
                if self.cursor_x < len(current_line):
                    # Delete character at cursor
                    self.lines[self.cursor_y] = current_line[:self.cursor_x] + current_line[self.cursor_x + 1:]
                    self.modified = True
                elif self.cursor_y < len(self.lines) - 1:
                    # Join with next line
                    next_line = self.lines[self.cursor_y + 1]
                    self.lines[self.cursor_y] = current_line + next_line
                    self.lines.pop(self.cursor_y + 1)
                    self.modified = True
                return True
                
            elif key == 23:  # Ctrl+W - save file
                self.save_file()
                return True
                
            elif key == 21:  # Ctrl+U - undo
                self._undo()
                return True
                
            elif key == 18:  # Ctrl+R - redo
                self._redo()
                return True
                
            elif key == 6:  # Ctrl+F - find
                self.mode = EditorMode.SEARCH
                self.search_buffer = ""
                self.search_results = []
                self.current_search_idx = 0
                return True
                
            elif key == 14:  # Ctrl+N - next search result
                self._next_search_result()
                return True
                
            elif key == 16:  # Ctrl+P - previous search result
                self._prev_search_result()
                return True
                
            elif key == 15:  # Ctrl+O - open file
                # Not implemented in this simple version
                self.status_message = "Open file not implemented in this version"
                return True
                
            elif key == 17:  # Ctrl+Q - quit
                # Check for unsaved changes
                if self.modified:
                    self.status_message = "Unsaved changes. Use :q! to force quit or :w to save."
                    return True
                return False  # Exit editor
                
            elif key == 1:  # Ctrl+A - select all (not implemented)
                self.status_message = "Select all not implemented in this version"
                return True
                
            elif key == 3:  # Ctrl+C - copy (not implemented)
                self.status_message = "Copy not implemented in this version"
                return True
                
            elif key == 22:  # Ctrl+V - paste (not implemented)
                self.status_message = "Paste not implemented in this version"
                return True
                
            elif key == 24:  # Ctrl+X - cut (not implemented)
                self.status_message = "Cut not implemented in this version"
                return True
                
            elif key == 8:  # Ctrl+H - show help
                self.mode = EditorMode.HELP
                return True
                
            elif key == 20:  # Ctrl+T - toggle syntax highlighting
                self.syntax_highlighting = not self.syntax_highlighting
                self.status_message = f"Syntax highlighting: {'on' if self.syntax_highlighting else 'off'}"
                return True
                
            elif key == 12:  # Ctrl+L - toggle line numbers
                self.line_numbers = not self.line_numbers
                self.status_message = f"Line numbers: {'on' if self.line_numbers else 'off'}"
                return True
                
            elif key == 19:  # Ctrl+S - save
                self.save_file()
                return True
                
            elif key == 5:  # Ctrl+E - show suggestions
                self._show_suggestions()
                return True
                
            elif 32 <= key <= 126:  # Printable ASCII characters
                # Save current state for undo
                self._save_undo_state()
                
                # Insert character at cursor
                char = chr(key)
                current_line = self.lines[self.cursor_y]
                self.lines[self.cursor_y] = current_line[:self.cursor_x] + char + current_line[self.cursor_x:]
                self.cursor_x += 1
                self.modified = True
                
                # Auto-show suggestions after certain characters
                if char in '.(' and self.agent:
                    self._show_suggestions()
                    
                return True
                
        elif self.mode == EditorMode.COMMAND:
            # Handle command mode keys
            if key == 27:  # ESC
                # Return to normal mode
                self.mode = EditorMode.NORMAL
                return True
                
            elif key == 10:  # Enter
                # Execute command
                result = self._execute_command(self.command_buffer)
                self.mode = EditorMode.NORMAL
                return result
                
            elif key == 127 or key == curses.KEY_BACKSPACE:  # Backspace
                # Delete character before cursor
                if self.command_buffer:
                    self.command_buffer = self.command_buffer[:-1]
                return True
                
            elif 32 <= key <= 126:  # Printable ASCII characters
                # Add character to command buffer
                self.command_buffer += chr(key)
                return True
                
        elif self.mode == EditorMode.SEARCH:
            # Handle search mode keys
            if key == 27:  # ESC
                # Return to normal mode
                self.mode = EditorMode.NORMAL
                return True
                
            elif key == 10:  # Enter
                # Execute search
                self._execute_search(self.search_buffer)
                self.mode = EditorMode.NORMAL
                return True
                
            elif key == 127 or key == curses.KEY_BACKSPACE:  # Backspace
                # Delete character before cursor
                if self.search_buffer:
                    self.search_buffer = self.search_buffer[:-1]
                    # Update search results as we type
                    self._execute_search(self.search_buffer)
                return True
                
            elif 32 <= key <= 126:  # Printable ASCII characters
                # Add character to search buffer
                self.search_buffer += chr(key)
                # Update search results as we type
                self._execute_search(self.search_buffer)
                return True
                
        return True
        
    def _execute_command(self, command):
        """Execute a command entered in command mode"""
        if not command:
            return True
            
        if command == 'q':
            # Quit
            if self.modified:
                self.status_message = "Unsaved changes. Use :q! to force quit or :w to save."
                return True
            return False  # Exit editor
            
        elif command == 'q!':
            # Force quit
            return False  # Exit editor
            
        elif command == 'w':
            # Save
            self.save_file()
            return True
            
        elif command == 'wq':
            # Save and quit
            if self.save_file():
                return False  # Exit editor
            return True
            
        elif command.startswith('set '):
            # Set editor option
            option = command[4:].strip()
            if option == 'syntax':
                self.syntax_highlighting = True
                self.status_message = "Syntax highlighting enabled"
            elif option == 'nosyntax':
                self.syntax_highlighting = False
                self.status_message = "Syntax highlighting disabled"
            elif option == 'number':
                self.line_numbers = True
                self.status_message = "Line numbers enabled"
            elif option == 'nonumber':
                self.line_numbers = False
                self.status_message = "Line numbers disabled"
            elif option.startswith('tabwidth='):
                try:
                    width = int(option.split('=')[1])
                    if 1 <= width <= 8:
                        self.tab_width = width
                        self.status_message = f"Tab width set to {width}"
                    else:
                        self.status_message = "Tab width must be between 1 and 8"
                except ValueError:
                    self.status_message = "Invalid tab width"
            elif option == 'ai' or option == 'autoindent':
                self.auto_indent = True
                self.status_message = "Auto-indent enabled"
            elif option == 'noai' or option == 'noautoindent':
                self.auto_indent = False
                self.status_message = "Auto-indent disabled"
            else:
                self.status_message = f"Unknown option: {option}"
            return True
            
        elif command.startswith('goto '):
            # Go to line
            try:
                line_num = int(command.split(' ')[1]) - 1
                if 0 <= line_num < len(self.lines):
                    self.cursor_y = line_num
                    self.cursor_x = 0
                    # Ensure line is visible
                    if self.cursor_y < self.scroll_y:
                        self.scroll_y = self.cursor_y
                    elif self.cursor_y >= self.scroll_y + curses.LINES - 3:
                        self.scroll_y = self.cursor_y - curses.LINES + 4
                    self.status_message = f"Moved to line {line_num + 1}"
                else:
                    self.status_message = f"Line {line_num + 1} out of range"
            except ValueError:
                self.status_message = "Invalid line number"
            return True
            
        elif command == 'help':
            # Show help
            self.mode = EditorMode.HELP
            return True
            
        elif command.startswith('analyze'):
            # Analyze code using the agent
            if self.agent:
                self.status_message = "Analyzing code..."
                # This would be implemented to call the agent's code analysis
                # For now, just a placeholder
                self.status_message = "Code analysis not implemented in this version"
            else:
                self.status_message = "Agent not available for code analysis"
            return True
            
        else:
            self.status_message = f"Unknown command: {command}"
            return True
            
    def _execute_search(self, search_text):
        """Execute a search and highlight results"""
        if not search_text:
            self.search_results = []
            return
            
        # Find all occurrences of search text
        self.search_results = []
        for i, line in enumerate(self.lines):
            start = 0
            while True:
                pos = line.find(search_text, start)
                if pos == -1:
                    break
                self.search_results.append((i, pos))
                start = pos + 1
                
        # Reset current search index
        self.current_search_idx = 0
        
        # Jump to first result if found
        if self.search_results:
            self._goto_search_result(0)
            
    def _goto_search_result(self, idx):
        """Go to a specific search result"""
        if not self.search_results or idx < 0 or idx >= len(self.search_results):
            return
            
        # Update current index
        self.current_search_idx = idx
        
        # Get line and column of result
        line, col = self.search_results[idx]
        
        # Move cursor to result
        self.cursor_y = line
        self.cursor_x = col
        
        # Ensure result is visible
        if self.cursor_y < self.scroll_y:
            self.scroll_y = self.cursor_y
        elif self.cursor_y >= self.scroll_y + curses.LINES - 3:
            self.scroll_y = self.cursor_y - curses.LINES + 4
            
    def _next_search_result(self):
        """Go to next search result"""
        if not self.search_results:
            return
            
        # Increment index with wraparound
        next_idx = (self.current_search_idx + 1) % len(self.search_results)
        self._goto_search_result(next_idx)
        
    def _prev_search_result(self):
        """Go to previous search result"""
        if not self.search_results:
            return
            
        # Decrement index with wraparound
        prev_idx = (self.current_search_idx - 1) % len(self.search_results)
        self._goto_search_result(prev_idx)
        
    def _save_undo_state(self):
        """Save current state for undo"""
        # Save current state
        state = {
            'lines': self.lines.copy(),
            'cursor_y': self.cursor_y,
            'cursor_x': self.cursor_x,
            'scroll_y': self.scroll_y
        }
        
        # Add to undo stack
        self.undo_stack.append(state)
        
        # Clear redo stack
        self.redo_stack = []
        
    def _undo(self):
        """Undo last change"""
        if not self.undo_stack:
            self.status_message = "Nothing to undo"
            return
            
        # Save current state for redo
        state = {
            'lines': self.lines.copy(),
            'cursor_y': self.cursor_y,
            'cursor_x': self.cursor_x,
            'scroll_y': self.scroll_y
        }
        self.redo_stack.append(state)
        
        # Restore previous state
        state = self.undo_stack.pop()
        self.lines = state['lines']
        self.cursor_y = state['cursor_y']
        self.cursor_x = state['cursor_x']
        self.scroll_y = state['scroll_y']
        
        self.status_message = "Undo"
        
    def _redo(self):
        """Redo last undone change"""
        if not self.redo_stack:
            self.status_message = "Nothing to redo"
            return
            
        # Save current state for undo
        state = {
            'lines': self.lines.copy(),
            'cursor_y': self.cursor_y,
            'cursor_x': self.cursor_x,
            'scroll_y': self.scroll_y
        }
        self.undo_stack.append(state)
        
        # Restore redo state
        state = self.redo_stack.pop()
        self.lines = state['lines']
        self.cursor_y = state['cursor_y']
        self.cursor_x = state['cursor_x']
        self.scroll_y = state['scroll_y']
        
        self.status_message = "Redo"
        
    def _show_suggestions(self):
        """Show auto-completion suggestions"""
        if not self.agent:
            self.status_message = "Agent not available for suggestions"
            return
            
        # Get current line and cursor position
        line = self.lines[self.cursor_y]
        pos = self.cursor_x
        
        # Extract context for suggestions
        context = line[:pos]
        
        # This would call the agent to get suggestions
        # For now, just use some dummy suggestions
        if context.endswith('.'):
            # Method suggestions
            self.suggestions = ['method1()', 'method2()', 'property', 'attribute']
        elif context.endswith('('):
            # Parameter suggestions
            self.suggestions = ['param1', 'param2', 'param3']
        else:
            # Variable/function suggestions
            self.suggestions = ['variable', 'function()', 'class', 'if', 'for', 'while']
            
        if self.suggestions:
            self.showing_suggestions = True
            self.selected_suggestion = 0
        else:
            self.showing_suggestions = False
            self.status_message = "No suggestions available"
            
    def _accept_suggestion(self):
        """Accept the currently selected suggestion"""
        if not self.showing_suggestions or not self.suggestions:
            return
            
        # Get the selected suggestion
        suggestion = self.suggestions[self.selected_suggestion]
        
        # Save current state for undo
        self._save_undo_state()
        
        # Get current line and cursor position
        line = self.lines[self.cursor_y]
        pos = self.cursor_x
        
        # Find the start of the current word
        word_start = pos
        while word_start > 0 and line[word_start - 1].isalnum() or line[word_start - 1] == '_':
            word_start -= 1
            
        # Replace current word with suggestion
        self.lines[self.cursor_y] = line[:word_start] + suggestion + line[pos:]
        self.cursor_x = word_start + len(suggestion)
        self.modified = True
        
        # Hide suggestions
        self.showing_suggestions = False
        
    def _get_help_text(self):
        """Get the help text for the editor"""
        return [
            "Keyboard Shortcuts:",
            "",
            "Navigation:",
            "  Arrow keys    - Move cursor",
            "  Home/End      - Start/end of line",
            "  Page Up/Down  - Move up/down one page",
            "",
            "Editing:",
            "  Enter         - Insert new line",
            "  Tab           - Insert spaces",
            "  Backspace     - Delete character before cursor",
            "  Delete        - Delete character at cursor",
            "",
            "File Operations:",
            "  Ctrl+S        - Save file",
            "  Ctrl+O        - Open file (not implemented)",
            "  Ctrl+Q        - Quit (with confirmation if unsaved changes)",
            "",
            "Search:",
            "  Ctrl+F        - Find text",
            "  Ctrl+N        - Next search result",
            "  Ctrl+P        - Previous search result",
            "",
            "Undo/Redo:",
            "  Ctrl+U        - Undo",
            "  Ctrl+R        - Redo",
            "",
            "Display Options:",
            "  Ctrl+T        - Toggle syntax highlighting",
            "  Ctrl+L        - Toggle line numbers",
            "",
            "Command Mode (press ':'):",
            "  :w            - Save file",
            "  :q            - Quit (if no unsaved changes)",
            "  :q!           - Force quit (discard changes)",
            "  :wq           - Save and quit",
            "  :set syntax   - Enable syntax highlighting",
            "  :set nosyntax - Disable syntax highlighting",
            "  :set number   - Enable line numbers",
            "  :set nonumber - Disable line numbers",
            "  :goto N       - Go to line N",
            "  :help         - Show this help",
            "",
            "Agent Integration:",
            "  Ctrl+E        - Show code suggestions",
            "  :analyze      - Analyze code with agent (not implemented)",
            "",
            "Press ESC to close this help window"
        ]

def main():
    """Main function to run the editor"""
    if len(sys.argv) < 2:
        print("Usage: python editor.py <file_path>")
        return
        
    file_path = sys.argv[1]
    editor = Editor(file_path)
    
    try:
        curses.wrapper(editor.run)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        pass

if __name__ == "__main__":
    main()
