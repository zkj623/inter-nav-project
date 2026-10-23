"""Nonblocking terminal editing; space cancels even while entering a target."""
import os
import select
import sys
import termios
import tty

MENU = '\n[o]观察 [n]点导航 [e]寻找物体 [p]暂停 [r]继续 [m]停止自主输出 [空格]取消 [q]退出\n'


class Keyboard:
    def __init__(self):
        if not sys.stdin.isatty():
            raise ValueError('Interactive mode needs a terminal (SSH with a TTY)')
        self.fd = sys.stdin.fileno()
        self.previous = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.editing = None
        self.text = ''
        print(MENU, flush=True)

    def poll(self):
        if not select.select([self.fd], [], [], 0)[0]:
            return None
        key = os.read(self.fd, 1).decode('utf-8', errors='ignore')
        if key in ('\x03', '\x04') or not key:
            return ('q', None)
        # Goal entry uses comma, so space is unambiguously stop everywhere.
        if key == ' ':
            self.editing, self.text = None, ''
            print('\n停止并取消任务', flush=True)
            return ('stop', None)
        if key == '\x1b':
            self.editing, self.text = None, ''
            print('\n取消输入', flush=True)
            return None
        if self.editing:
            if key in ('\n', '\r'):
                operation, value = self.editing, self.text.strip()
                self.editing, self.text = None, ''
                print('', flush=True)
                return (operation, value)
            if key in ('\x7f', '\b'):
                self.text = self.text[:-1]
                print('\b \b', end='', flush=True)
            elif key.isprintable():
                self.text += key
                print(key, end='', flush=True)
            return None
        if key in ('n', 'e'):
            self.editing = 'navigate' if key == 'n' else 'explore'
            print('\n先停止当前任务。输入地图 x,y：' if key == 'n' else
                  '\n先停止当前任务。输入英文目标名（下划线表示空格）：', end='', flush=True)
            return ('prepare', None)
        if key in ('o', 'p', 'r', 'm', 'q'):
            return (key, None)
        return None

    def close(self):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.previous)
