#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on September 22, 2025, at 09:21
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
# PsychoPy version compatibility: some installs do not have plugins.activatePlugins().
# This keeps Builder-generated plugin activation when available, but avoids crashing
# on older/different PsychoPy installs.
try:
    from psychopy import plugins
    if hasattr(plugins, 'activatePlugins'):
        plugins.activatePlugins()
except Exception:
    pass
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors

# PsychoPy version compatibility: newer Builder scripts import layout/environmenttools/priority,
# but older installs may not have one or more of these. These are not used to change
# the task logic or output; the guards only prevent startup crashes across versions.
try:
    from psychopy import layout
except Exception:
    layout = None

try:
    from psychopy.tools import environmenttools
except Exception:
    class _EnvironmentToolsFallback:
        @staticmethod
        def setExecEnvironment(globals_dict):
            return None
    environmenttools = _EnvironmentToolsFallback()

try:
    from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                    STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)
except Exception:
    from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                    STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)
    class _PriorityFallback:
        CRITICAL = 'critical'
        LOW = 'low'
    priority = _PriorityFallback()

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'RewardGoNoGo'  # from the Builder filename that created this script
expInfo = {
    'mriMode': 'Scan',
    'runOrder': 'A',
    'hand': 'right',
    'session': '001',
    'stimType': 'c',
    'participant': 'Chein-LITe.#_P####',
    'colorVersion': '1Striped',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}

# Resume/relaunch controls. These are intentionally NOT stored in expInfo,
# so the research output columns stay the same as the original task.
START_ROUND = 1
RESUME_FILE_LABEL = ''


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    global START_ROUND, RESUME_FILE_LABEL
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show start round in the dialog, but remove it before data saving
    # so the output columns stay the same as the original task.
    expInfo['startRound'] = '1'
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # pull resume setting out of expInfo before ExperimentHandler sees it
    startRoundRaw = expInfo.pop('startRound', '1')
    try:
        START_ROUND = int(startRoundRaw)
    except (TypeError, ValueError):
        START_ROUND = 1
    if START_ROUND < 1:
        START_ROUND = 1
    if START_ROUND > 4:
        START_ROUND = 4
    if START_ROUND > 1:
        RESUME_FILE_LABEL = '_RESUME_FROM_ROUND_%d' % START_ROUND
    else:
        RESUME_FILE_LABEL = ''
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    # Add a filename-only resume label when relaunching from a later round.
    # This does NOT add any new data columns.
    filename = u'data' + os.sep + '%s_%s%s' %(expInfo['participant'], expInfo['date'], RESUME_FILE_LABEL)
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    # Use overwrite for this launch's own files so round-level autosaves update
    # the same CSV/psydat instead of creating name-collision copies.
    try:
        thisExp = data.ExperimentHandler(
            name=expName, version='',
            extraInfo=expInfo, runtimeInfo=None,
            originPath='C:\\Users\\Public\\LAB PROJECTS\\Chein-Lab\\LITe\\planetGoNoGo\\PlanetGoNoGo_scan_update0613_lastrun.py',
            savePickle=True, saveWideText=True,
            dataFileName=dataDir + os.sep + filename, sortColumns='time',
            fileCollisionMethod='overwrite'
        )
    except TypeError:
        # Older PsychoPy versions may not accept fileCollisionMethod here.
        # The custom saveData() below still saves to one stable file per launch.
        thisExp = data.ExperimentHandler(
            name=expName, version='',
            extraInfo=expInfo, runtimeInfo=None,
            originPath='C:\\Users\\Public\\LAB PROJECTS\\Chein-Lab\\LITe\\planetGoNoGo\\PlanetGoNoGo_scan_update0613_lastrun.py',
            savePickle=True, saveWideText=True,
            dataFileName=dataDir + os.sep + filename, sortColumns='time'
        )
    # Some older PsychoPy versions do not support column priority settings.
    # Skipping these does not add/remove data columns; it only affects column ordering metadata.
    try:
        thisExp.setPriority('thisRow.t', priority.CRITICAL)
        thisExp.setPriority('expName', priority.LOW)
    except Exception:
        pass
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.DEBUG)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.DEBUG)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[2560, 1440], fullscr=True, screen=1,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[-1,-1,-1], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='deg'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [-1,-1,-1]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'deg'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Welcome" ---
    welcomeScreen = visual.TextStim(win=win, name='welcomeScreen',
        text='Welcome to the planets game!',
        font='Arial',
        pos=[0, 0], height=2, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    advanceScreen = keyboard.Keyboard()
    polygon = visual.Rect(
        win=win, name='polygon',
        width=[2, 2][0], height=[2, 2][1],
        ori=0, pos=[-15, 12], anchor='center',
        lineWidth=1,     colorSpace='rgb',  lineColor=[1,1,1], fillColor=[1,1,1],
        opacity=1, depth=-2.0, interpolate=True)
    
    # --- Initialize components for Routine "text1" ---
    advanceScreen_2 = keyboard.Keyboard()
    directionText_1 = visual.TextStim(win=win, name='directionText_1',
        text='In this game, you will be traveling through different galaxies in space. In these galaxies, you will see planets or moons.',
        font='Arial',
        pos=[0, 0], height=1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    # Run 'Begin Experiment' code from setConditionsFile
    colorTrialsFile = os.path.join('conditions', 'RewardGoNoGo_Version%s.xlsx' % expInfo['colorVersion'])
    logging.exp("Using trial color file: %s" % colorTrialsFile)
    runOrderFile = os.path.join('conditions', 'GalaxyValues%s.xlsx' % expInfo['runOrder'])
    logging.exp("Using run order file: %s" % runOrderFile)
    
    # --- Initialize components for Routine "text2" ---
    advanceScreen_3 = keyboard.Keyboard()
    directionText_2 = visual.TextStim(win=win, name='directionText_2',
        text='Moons have craters. When you see a moon, do NOTHING. Here are some examples of the moons you may see:',
        font='Arial',
        pos=[0, 0], height=1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "image1" ---
    advanceScreen_4 = keyboard.Keyboard()
    moonExampleImage = visual.ImageStim(
        win=win,
        name='moonExampleImage', 
        image='instructionimages/gng_moons.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(30, 30),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "text3" ---
    advanceScreen_5 = keyboard.Keyboard()
    directionsText_3 = visual.TextStim(win=win, name='directionsText_3',
        text='Planets have stripes. Here are some examples of planets you may see:',
        font='Arial',
        pos=[0, 0], height=1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "image2" ---
    advanceScreen_6 = keyboard.Keyboard()
    planetExampleImage = visual.ImageStim(
        win=win,
        name='planetExampleImage', 
        image='instructionimages/gng_planets.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(30, 30),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "text4" ---
    advanceScreen_7 = keyboard.Keyboard()
    directionText_3 = visual.TextStim(win=win, name='directionText_3',
        text='When you see a planet, click your pointer finger as quickly as possible. Remember to press the button all the way down, and then let it go right after.',
        font='Arial',
        pos=[0, 0], height=1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "text5" ---
    advanceScreen_8 = keyboard.Keyboard()
    directionText_4 = visual.TextStim(win=win, name='directionText_4',
        text="You're going to see four different galaxies. Before each galaxy, you're going to see either one star or two stars.",
        font='Arial',
        pos=[0, 0], height=1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "image3" ---
    advanceScreen_9 = keyboard.Keyboard()
    cueExampleImage_1 = visual.ImageStim(
        win=win,
        name='cueExampleImage_1', 
        image='instructionimages/gng_starcue1.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(30, 30),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "text6" ---
    advanceScreen_10 = keyboard.Keyboard()
    directionText_5 = visual.TextStim(win=win, name='directionText_5',
        text="If you see one star before you enter a galaxy, that means you can win 200 points when you click a planet.\n\nBut when you don't click a planet, or you click a moon, you lose 100 points.",
        font='Arial',
        pos=[0, 0], height=1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "image4" ---
    advanceScreen_11 = keyboard.Keyboard()
    cueExampleImage_2 = visual.ImageStim(
        win=win,
        name='cueExampleImage_2', 
        image='instructionimages/gng_starcue2.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(30, 30),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "text7" ---
    advanceScreen_12 = keyboard.Keyboard()
    directionText_6 = visual.TextStim(win=win, name='directionText_6',
        text="If you see two stars before you enter a galaxy, that means you can win 1000 points when you click a planet.\n\nBut when you don't click a planet, or you click a moon, you lose 500 points.",
        font='Arial',
        pos=[0, 0], height=1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "text8" ---
    advanceScreen_13 = keyboard.Keyboard()
    directionText_7 = visual.TextStim(win=win, name='directionText_7',
        text="At the end of each galaxy, you'll see how many points you earned, and between each galaxy you'll get to take a short break. \n\nRemember, use your pointer finger, and click ONLY for planets. \n\nDo you have any questions?",
        font='Arial',
        pos=[0, 0], height=1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "trigger" ---
    waitForTrigger = visual.TextStim(win=win, name='waitForTrigger',
        text='Get Ready!',
        font='Arial',
        pos=[0, 0], height=1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    receiveTrigger = keyboard.Keyboard()
    
    # --- Initialize components for Routine "waitForScanner" ---
    # Run 'Begin Experiment' code from wait_for_scanner_code
    fmriClock = core.Clock()
    #trigger = 'usb'
    #trigger = 'parallel'
    #if trigger == 'parallel':
    #    from psychopy.contrib import parallel as winioport
    #    #from psychopy import parallel
    #elif trigger == 'usb':
    #    from psychopy.hardware.emulator import launchScan
        #
        # settings for launchScan:
    #    MR_settings = { 
    #        'TR': 2.000, # duration (sec) per volume
    #        'volumes': 210, # number of whole-brain 3D volumes / frames
    #        'sync': 'equal', # character to use as the sync timing event; assumed to come at start of a volume
    #        'skip': 0, # number of volumes lacking a sync pulse at start of scan (for T1 stabilization)
    #        }
    
    # for GSR sync
    #import serial
    #class HvdSerial(serial.Serial):
    #    def writeUSB(self,dest,val):
    #        self.write(chr(dest))
    #        self.write(chr(val))
    
    #try:
    #    ser = HvdSerial('/dev/tty.usbmodem12341',timeout=1)
        # Zero Out Parallel & BNC
    #    ser.writeUSB(0,0)
    #    core.wait(.3)
    #    ser.writeUSB(1,0)
    #    core.wait(.3)
    #except serial.SerialException:
    #    ser = None
    #    logging.warn('No Serial Device Found.')
    
    # --- Initialize components for Routine "trial_cue" ---
    # Run 'Begin Experiment' code from code
    # set border color
    borderColor='black'
    valueCondition=None
    valueConditions=None
    Cue2 = visual.ImageStim(
        win=win,
        name='Cue2', 
        image='default.png', mask=None, anchor='center',
        ori=0, pos=[0, 0], size=[12.5, 10],
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=False, flipVert=False,
        texRes=128, interpolate=True, depth=-2.0)
    frame1 = visual.Rect(
        win=win, name='frame1',
        width=[18, 18][0], height=[18, 18][1],
        ori=0, pos=[0, 0], anchor='center',
        lineWidth=1,     colorSpace='rgb',  lineColor='white', fillColor=None,
        opacity=1, depth=-3.0, interpolate=True)
    frame2 = visual.Rect(
        win=win, name='frame2',
        width=[19, 19][0], height=[19, 19][1],
        ori=0, pos=[0, 0], anchor='center',
        lineWidth=1,     colorSpace='rgb',  lineColor=[1,1,1], fillColor=None,
        opacity=1, depth=-4.0, interpolate=True)
    # Run 'Begin Experiment' code from generateJitterList
    allISIs=[3.5, 2.5, 3, 2.5, 1.5, 2, 1.5, 2.5]
    
    import copy, random
    # Run 'Begin Experiment' code from setStimRotations
    rotations=[45, 90, 135, 180, 225, 270, 315, 360]
    
    import copy, random
    drawCircle = visual.Polygon(
        win=win, name='drawCircle',
        edges=90, size=[16.5, 16.5],
        ori=0, pos=[0, 0], anchor='center',
        lineWidth=5,     colorSpace='rgb',  lineColor=[1,1,1], fillColor=None,
        opacity=1, depth=-7.0, interpolate=True)
    # Run 'Begin Experiment' code from blockTiming
    import time
    expInfo['expStartTime'] = time.ctime()
    
    BLOCK_DURATION = 10
    blockClock = core.Clock()
    
    # --- Initialize components for Routine "fixation" ---
    fixation1 = visual.TextStim(win=win, name='fixation1',
        text='',
        font='Arial',
        pos=[0, 0], height=1.5, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    fixationFrame1 = visual.Rect(
        win=win, name='fixationFrame1',
        width=[18, 18][0], height=[18, 18][1],
        ori=0, pos=[0, 0], anchor='center',
        lineWidth=1,     colorSpace='rgb',  lineColor=[1,1,1], fillColor=None,
        opacity=1, depth=-2.0, interpolate=True)
    fixationFrame2 = visual.Rect(
        win=win, name='fixationFrame2',
        width=[19, 19][0], height=[19, 19][1],
        ori=0, pos=[0, 0], anchor='center',
        lineWidth=1,     colorSpace='rgb',  lineColor=[1,1,1], fillColor=None,
        opacity=1, depth=-3.0, interpolate=True)
    
    # --- Initialize components for Routine "trial" ---
    # Run 'Begin Experiment' code from setTargetColors
    targetType=None
    prepot=None
    # Run 'Begin Experiment' code from setKeyResponses
    # set border color
    borderColor='black'
    
    # Assumes "first key" or "last key" was chosen for store.
    # List conversion (to single key) happens immediately
    # when component finishes.
    def check_correct(keys,correct):
        if len(keys):  # At least 1 response
            if keys == correct:
                return True
            else:
                return False
        else:  # No response
            if correct == None:
                return True
            else:
                return False
    
    # set outcome values
    outcomeNumber='0'
    outcomeText='0'
    
    
    
    trialFrame = visual.Rect(
        win=win, name='trialFrame',
        width=[18, 18][0], height=[18, 18][1],
        ori=0, pos=[0, 0], anchor='center',
        lineWidth=1,     colorSpace='rgb',  lineColor='white', fillColor=None,
        opacity=1, depth=-4.0, interpolate=True)
    trialFrame2 = visual.Rect(
        win=win, name='trialFrame2',
        width=[19, 19][0], height=[19, 19][1],
        ori=0, pos=[0, 0], anchor='center',
        lineWidth=1,     colorSpace='rgb',  lineColor='black', fillColor=None,
        opacity=1, depth=-5.0, interpolate=True)
    Target = visual.ImageStim(
        win=win,
        name='Target', 
        image='default.png', mask=None, anchor='center',
        ori=1.0, pos=[0, 0], size=[15, 15],
        color=[1,1,1], colorSpace='rgb', opacity=1,
        flipHoriz=True, flipVert=False,
        texRes=256, interpolate=True, depth=-6.0)
    targetResp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "jitter" ---
    jitteredISI = visual.TextStim(win=win, name='jitteredISI',
        text='+',
        font='Arial',
        pos=[0, 0], height=1.5, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    jitterFrame1 = visual.Rect(
        win=win, name='jitterFrame1',
        width=[18, 18][0], height=[18, 18][1],
        ori=0, pos=[0, 0], anchor='center',
        lineWidth=1,     colorSpace='rgb',  lineColor=[1,1,1], fillColor=None,
        opacity=1, depth=-1.0, interpolate=True)
    jitterFrame2 = visual.Rect(
        win=win, name='jitterFrame2',
        width=[19, 19][0], height=[19, 19][1],
        ori=0, pos=[0, 0], anchor='center',
        lineWidth=1,     colorSpace='rgb',  lineColor=[1,1,1], fillColor=None,
        opacity=1, depth=-2.0, interpolate=True)
    lateResp = keyboard.Keyboard()
    # Run 'Begin Experiment' code from recordRT
    RT=None
    goRT=None
    
    
    # --- Initialize components for Routine "feedback" ---
    # Run 'Begin Experiment' code from rewardFrames
    # set reward text color
    rewardColor='white'
    reward = visual.TextStim(win=win, name='reward',
        text='',
        font='Arial',
        pos=[0, 0], height=2, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    rewardFrame1 = visual.Rect(
        win=win, name='rewardFrame1',
        width=[18, 18][0], height=[18, 18][1],
        ori=0, pos=[0, 0], anchor='center',
        lineWidth=1,     colorSpace='rgb',  lineColor=[1,1,1], fillColor=None,
        opacity=1, depth=-2.0, interpolate=True)
    rewardFrame2 = visual.Rect(
        win=win, name='rewardFrame2',
        width=[19, 19][0], height=[19, 19][1],
        ori=0, pos=[0, 0], anchor='center',
        lineWidth=1,     colorSpace='rgb',  lineColor=[1,1,1], fillColor=None,
        opacity=1, depth=-3.0, interpolate=True)
    
    # --- Initialize components for Routine "endBlock" ---
    breakFixation = visual.TextStim(win=win, name='breakFixation',
        text='+',
        font='Arial',
        pos=[0, 0], height=1.5, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "runEnd" ---
    breakText = visual.TextStim(win=win, name='breakText',
        text='You finished this round!',
        font='Arial',
        pos=[0, 0], height=1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    SpaceToEndRun = keyboard.Keyboard()
    
    # --- Initialize components for Routine "End" ---
    endTask = visual.TextStim(win=win, name='endTask',
        text='Congratulations! You finished the game!',
        font='Arial',
        pos=[0, 0], height=1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    SpaceToExit = keyboard.Keyboard()
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "Welcome" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Welcome.started', globalClock.getTime())
    advanceScreen.keys = []
    advanceScreen.rt = []
    _advanceScreen_allKeys = []
    # keep track of which components have finished
    WelcomeComponents = [welcomeScreen, advanceScreen, polygon]
    for thisComponent in WelcomeComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Welcome" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *welcomeScreen* updates
        
        # if welcomeScreen is starting this frame...
        if welcomeScreen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcomeScreen.frameNStart = frameN  # exact frame index
            welcomeScreen.tStart = t  # local t and not account for scr refresh
            welcomeScreen.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcomeScreen, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcomeScreen.started')
            # update status
            welcomeScreen.status = STARTED
            welcomeScreen.setAutoDraw(True)
        
        # if welcomeScreen is active this frame...
        if welcomeScreen.status == STARTED:
            # update params
            pass
        
        # *advanceScreen* updates
        waitOnFlip = False
        
        # if advanceScreen is starting this frame...
        if advanceScreen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            advanceScreen.frameNStart = frameN  # exact frame index
            advanceScreen.tStart = t  # local t and not account for scr refresh
            advanceScreen.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(advanceScreen, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'advanceScreen.started')
            # update status
            advanceScreen.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(advanceScreen.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(advanceScreen.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if advanceScreen.status == STARTED and not waitOnFlip:
            theseKeys = advanceScreen.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
            _advanceScreen_allKeys.extend(theseKeys)
            if len(_advanceScreen_allKeys):
                advanceScreen.keys = _advanceScreen_allKeys[-1].name  # just the last key pressed
                advanceScreen.rt = _advanceScreen_allKeys[-1].rt
                advanceScreen.duration = _advanceScreen_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *polygon* updates
        
        # if polygon is starting this frame...
        if polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            polygon.frameNStart = frameN  # exact frame index
            polygon.tStart = t  # local t and not account for scr refresh
            polygon.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(polygon, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'polygon.started')
            # update status
            polygon.status = STARTED
            polygon.setAutoDraw(True)
        
        # if polygon is active this frame...
        if polygon.status == STARTED:
            # update params
            pass
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in WelcomeComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Welcome" ---
    for thisComponent in WelcomeComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Welcome.stopped', globalClock.getTime())
    # the Routine "Welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "text1" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('text1.started', globalClock.getTime())
    advanceScreen_2.keys = []
    advanceScreen_2.rt = []
    _advanceScreen_2_allKeys = []
    # keep track of which components have finished
    text1Components = [advanceScreen_2, directionText_1]
    for thisComponent in text1Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "text1" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *advanceScreen_2* updates
        waitOnFlip = False
        
        # if advanceScreen_2 is starting this frame...
        if advanceScreen_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            advanceScreen_2.frameNStart = frameN  # exact frame index
            advanceScreen_2.tStart = t  # local t and not account for scr refresh
            advanceScreen_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(advanceScreen_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'advanceScreen_2.started')
            # update status
            advanceScreen_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(advanceScreen_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(advanceScreen_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if advanceScreen_2.status == STARTED and not waitOnFlip:
            theseKeys = advanceScreen_2.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
            _advanceScreen_2_allKeys.extend(theseKeys)
            if len(_advanceScreen_2_allKeys):
                advanceScreen_2.keys = _advanceScreen_2_allKeys[-1].name  # just the last key pressed
                advanceScreen_2.rt = _advanceScreen_2_allKeys[-1].rt
                advanceScreen_2.duration = _advanceScreen_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *directionText_1* updates
        
        # if directionText_1 is starting this frame...
        if directionText_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            directionText_1.frameNStart = frameN  # exact frame index
            directionText_1.tStart = t  # local t and not account for scr refresh
            directionText_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(directionText_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'directionText_1.started')
            # update status
            directionText_1.status = STARTED
            directionText_1.setAutoDraw(True)
        
        # if directionText_1 is active this frame...
        if directionText_1.status == STARTED:
            # update params
            pass
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in text1Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "text1" ---
    for thisComponent in text1Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('text1.stopped', globalClock.getTime())
    # the Routine "text1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "text2" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('text2.started', globalClock.getTime())
    advanceScreen_3.keys = []
    advanceScreen_3.rt = []
    _advanceScreen_3_allKeys = []
    # keep track of which components have finished
    text2Components = [advanceScreen_3, directionText_2]
    for thisComponent in text2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "text2" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *advanceScreen_3* updates
        waitOnFlip = False
        
        # if advanceScreen_3 is starting this frame...
        if advanceScreen_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            advanceScreen_3.frameNStart = frameN  # exact frame index
            advanceScreen_3.tStart = t  # local t and not account for scr refresh
            advanceScreen_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(advanceScreen_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'advanceScreen_3.started')
            # update status
            advanceScreen_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(advanceScreen_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(advanceScreen_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if advanceScreen_3.status == STARTED and not waitOnFlip:
            theseKeys = advanceScreen_3.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
            _advanceScreen_3_allKeys.extend(theseKeys)
            if len(_advanceScreen_3_allKeys):
                advanceScreen_3.keys = _advanceScreen_3_allKeys[-1].name  # just the last key pressed
                advanceScreen_3.rt = _advanceScreen_3_allKeys[-1].rt
                advanceScreen_3.duration = _advanceScreen_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *directionText_2* updates
        
        # if directionText_2 is starting this frame...
        if directionText_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            directionText_2.frameNStart = frameN  # exact frame index
            directionText_2.tStart = t  # local t and not account for scr refresh
            directionText_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(directionText_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'directionText_2.started')
            # update status
            directionText_2.status = STARTED
            directionText_2.setAutoDraw(True)
        
        # if directionText_2 is active this frame...
        if directionText_2.status == STARTED:
            # update params
            pass
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in text2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "text2" ---
    for thisComponent in text2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('text2.stopped', globalClock.getTime())
    # the Routine "text2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "image1" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('image1.started', globalClock.getTime())
    advanceScreen_4.keys = []
    advanceScreen_4.rt = []
    _advanceScreen_4_allKeys = []
    # keep track of which components have finished
    image1Components = [advanceScreen_4, moonExampleImage]
    for thisComponent in image1Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "image1" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *advanceScreen_4* updates
        waitOnFlip = False
        
        # if advanceScreen_4 is starting this frame...
        if advanceScreen_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            advanceScreen_4.frameNStart = frameN  # exact frame index
            advanceScreen_4.tStart = t  # local t and not account for scr refresh
            advanceScreen_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(advanceScreen_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'advanceScreen_4.started')
            # update status
            advanceScreen_4.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(advanceScreen_4.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(advanceScreen_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if advanceScreen_4.status == STARTED and not waitOnFlip:
            theseKeys = advanceScreen_4.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
            _advanceScreen_4_allKeys.extend(theseKeys)
            if len(_advanceScreen_4_allKeys):
                advanceScreen_4.keys = _advanceScreen_4_allKeys[-1].name  # just the last key pressed
                advanceScreen_4.rt = _advanceScreen_4_allKeys[-1].rt
                advanceScreen_4.duration = _advanceScreen_4_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *moonExampleImage* updates
        
        # if moonExampleImage is starting this frame...
        if moonExampleImage.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            moonExampleImage.frameNStart = frameN  # exact frame index
            moonExampleImage.tStart = t  # local t and not account for scr refresh
            moonExampleImage.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(moonExampleImage, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'moonExampleImage.started')
            # update status
            moonExampleImage.status = STARTED
            moonExampleImage.setAutoDraw(True)
        
        # if moonExampleImage is active this frame...
        if moonExampleImage.status == STARTED:
            # update params
            pass
        
        # if moonExampleImage is stopping this frame...
        if moonExampleImage.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > moonExampleImage.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                moonExampleImage.tStop = t  # not accounting for scr refresh
                moonExampleImage.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'moonExampleImage.stopped')
                # update status
                moonExampleImage.status = FINISHED
                moonExampleImage.setAutoDraw(False)
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in image1Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "image1" ---
    for thisComponent in image1Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('image1.stopped', globalClock.getTime())
    # the Routine "image1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "text3" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('text3.started', globalClock.getTime())
    advanceScreen_5.keys = []
    advanceScreen_5.rt = []
    _advanceScreen_5_allKeys = []
    # keep track of which components have finished
    text3Components = [advanceScreen_5, directionsText_3]
    for thisComponent in text3Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "text3" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *advanceScreen_5* updates
        waitOnFlip = False
        
        # if advanceScreen_5 is starting this frame...
        if advanceScreen_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            advanceScreen_5.frameNStart = frameN  # exact frame index
            advanceScreen_5.tStart = t  # local t and not account for scr refresh
            advanceScreen_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(advanceScreen_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'advanceScreen_5.started')
            # update status
            advanceScreen_5.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(advanceScreen_5.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(advanceScreen_5.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if advanceScreen_5.status == STARTED and not waitOnFlip:
            theseKeys = advanceScreen_5.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
            _advanceScreen_5_allKeys.extend(theseKeys)
            if len(_advanceScreen_5_allKeys):
                advanceScreen_5.keys = _advanceScreen_5_allKeys[-1].name  # just the last key pressed
                advanceScreen_5.rt = _advanceScreen_5_allKeys[-1].rt
                advanceScreen_5.duration = _advanceScreen_5_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *directionsText_3* updates
        
        # if directionsText_3 is starting this frame...
        if directionsText_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            directionsText_3.frameNStart = frameN  # exact frame index
            directionsText_3.tStart = t  # local t and not account for scr refresh
            directionsText_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(directionsText_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'directionsText_3.started')
            # update status
            directionsText_3.status = STARTED
            directionsText_3.setAutoDraw(True)
        
        # if directionsText_3 is active this frame...
        if directionsText_3.status == STARTED:
            # update params
            pass
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in text3Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "text3" ---
    for thisComponent in text3Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('text3.stopped', globalClock.getTime())
    # the Routine "text3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "image2" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('image2.started', globalClock.getTime())
    advanceScreen_6.keys = []
    advanceScreen_6.rt = []
    _advanceScreen_6_allKeys = []
    # keep track of which components have finished
    image2Components = [advanceScreen_6, planetExampleImage]
    for thisComponent in image2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "image2" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *advanceScreen_6* updates
        waitOnFlip = False
        
        # if advanceScreen_6 is starting this frame...
        if advanceScreen_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            advanceScreen_6.frameNStart = frameN  # exact frame index
            advanceScreen_6.tStart = t  # local t and not account for scr refresh
            advanceScreen_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(advanceScreen_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'advanceScreen_6.started')
            # update status
            advanceScreen_6.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(advanceScreen_6.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(advanceScreen_6.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if advanceScreen_6.status == STARTED and not waitOnFlip:
            theseKeys = advanceScreen_6.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
            _advanceScreen_6_allKeys.extend(theseKeys)
            if len(_advanceScreen_6_allKeys):
                advanceScreen_6.keys = _advanceScreen_6_allKeys[-1].name  # just the last key pressed
                advanceScreen_6.rt = _advanceScreen_6_allKeys[-1].rt
                advanceScreen_6.duration = _advanceScreen_6_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *planetExampleImage* updates
        
        # if planetExampleImage is starting this frame...
        if planetExampleImage.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            planetExampleImage.frameNStart = frameN  # exact frame index
            planetExampleImage.tStart = t  # local t and not account for scr refresh
            planetExampleImage.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(planetExampleImage, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'planetExampleImage.started')
            # update status
            planetExampleImage.status = STARTED
            planetExampleImage.setAutoDraw(True)
        
        # if planetExampleImage is active this frame...
        if planetExampleImage.status == STARTED:
            # update params
            pass
        
        # if planetExampleImage is stopping this frame...
        if planetExampleImage.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > planetExampleImage.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                planetExampleImage.tStop = t  # not accounting for scr refresh
                planetExampleImage.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'planetExampleImage.stopped')
                # update status
                planetExampleImage.status = FINISHED
                planetExampleImage.setAutoDraw(False)
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in image2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "image2" ---
    for thisComponent in image2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('image2.stopped', globalClock.getTime())
    # the Routine "image2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "text4" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('text4.started', globalClock.getTime())
    advanceScreen_7.keys = []
    advanceScreen_7.rt = []
    _advanceScreen_7_allKeys = []
    # keep track of which components have finished
    text4Components = [advanceScreen_7, directionText_3]
    for thisComponent in text4Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "text4" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *advanceScreen_7* updates
        waitOnFlip = False
        
        # if advanceScreen_7 is starting this frame...
        if advanceScreen_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            advanceScreen_7.frameNStart = frameN  # exact frame index
            advanceScreen_7.tStart = t  # local t and not account for scr refresh
            advanceScreen_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(advanceScreen_7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'advanceScreen_7.started')
            # update status
            advanceScreen_7.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(advanceScreen_7.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(advanceScreen_7.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if advanceScreen_7.status == STARTED and not waitOnFlip:
            theseKeys = advanceScreen_7.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
            _advanceScreen_7_allKeys.extend(theseKeys)
            if len(_advanceScreen_7_allKeys):
                advanceScreen_7.keys = _advanceScreen_7_allKeys[-1].name  # just the last key pressed
                advanceScreen_7.rt = _advanceScreen_7_allKeys[-1].rt
                advanceScreen_7.duration = _advanceScreen_7_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *directionText_3* updates
        
        # if directionText_3 is starting this frame...
        if directionText_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            directionText_3.frameNStart = frameN  # exact frame index
            directionText_3.tStart = t  # local t and not account for scr refresh
            directionText_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(directionText_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'directionText_3.started')
            # update status
            directionText_3.status = STARTED
            directionText_3.setAutoDraw(True)
        
        # if directionText_3 is active this frame...
        if directionText_3.status == STARTED:
            # update params
            pass
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in text4Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "text4" ---
    for thisComponent in text4Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('text4.stopped', globalClock.getTime())
    # the Routine "text4" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "text5" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('text5.started', globalClock.getTime())
    advanceScreen_8.keys = []
    advanceScreen_8.rt = []
    _advanceScreen_8_allKeys = []
    # keep track of which components have finished
    text5Components = [advanceScreen_8, directionText_4]
    for thisComponent in text5Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "text5" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *advanceScreen_8* updates
        waitOnFlip = False
        
        # if advanceScreen_8 is starting this frame...
        if advanceScreen_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            advanceScreen_8.frameNStart = frameN  # exact frame index
            advanceScreen_8.tStart = t  # local t and not account for scr refresh
            advanceScreen_8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(advanceScreen_8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'advanceScreen_8.started')
            # update status
            advanceScreen_8.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(advanceScreen_8.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(advanceScreen_8.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if advanceScreen_8.status == STARTED and not waitOnFlip:
            theseKeys = advanceScreen_8.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
            _advanceScreen_8_allKeys.extend(theseKeys)
            if len(_advanceScreen_8_allKeys):
                advanceScreen_8.keys = _advanceScreen_8_allKeys[-1].name  # just the last key pressed
                advanceScreen_8.rt = _advanceScreen_8_allKeys[-1].rt
                advanceScreen_8.duration = _advanceScreen_8_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *directionText_4* updates
        
        # if directionText_4 is starting this frame...
        if directionText_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            directionText_4.frameNStart = frameN  # exact frame index
            directionText_4.tStart = t  # local t and not account for scr refresh
            directionText_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(directionText_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'directionText_4.started')
            # update status
            directionText_4.status = STARTED
            directionText_4.setAutoDraw(True)
        
        # if directionText_4 is active this frame...
        if directionText_4.status == STARTED:
            # update params
            pass
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in text5Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "text5" ---
    for thisComponent in text5Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('text5.stopped', globalClock.getTime())
    # the Routine "text5" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "image3" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('image3.started', globalClock.getTime())
    advanceScreen_9.keys = []
    advanceScreen_9.rt = []
    _advanceScreen_9_allKeys = []
    # keep track of which components have finished
    image3Components = [advanceScreen_9, cueExampleImage_1]
    for thisComponent in image3Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "image3" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *advanceScreen_9* updates
        waitOnFlip = False
        
        # if advanceScreen_9 is starting this frame...
        if advanceScreen_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            advanceScreen_9.frameNStart = frameN  # exact frame index
            advanceScreen_9.tStart = t  # local t and not account for scr refresh
            advanceScreen_9.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(advanceScreen_9, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'advanceScreen_9.started')
            # update status
            advanceScreen_9.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(advanceScreen_9.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(advanceScreen_9.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if advanceScreen_9.status == STARTED and not waitOnFlip:
            theseKeys = advanceScreen_9.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
            _advanceScreen_9_allKeys.extend(theseKeys)
            if len(_advanceScreen_9_allKeys):
                advanceScreen_9.keys = _advanceScreen_9_allKeys[-1].name  # just the last key pressed
                advanceScreen_9.rt = _advanceScreen_9_allKeys[-1].rt
                advanceScreen_9.duration = _advanceScreen_9_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *cueExampleImage_1* updates
        
        # if cueExampleImage_1 is starting this frame...
        if cueExampleImage_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cueExampleImage_1.frameNStart = frameN  # exact frame index
            cueExampleImage_1.tStart = t  # local t and not account for scr refresh
            cueExampleImage_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cueExampleImage_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'cueExampleImage_1.started')
            # update status
            cueExampleImage_1.status = STARTED
            cueExampleImage_1.setAutoDraw(True)
        
        # if cueExampleImage_1 is active this frame...
        if cueExampleImage_1.status == STARTED:
            # update params
            pass
        
        # if cueExampleImage_1 is stopping this frame...
        if cueExampleImage_1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > cueExampleImage_1.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                cueExampleImage_1.tStop = t  # not accounting for scr refresh
                cueExampleImage_1.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cueExampleImage_1.stopped')
                # update status
                cueExampleImage_1.status = FINISHED
                cueExampleImage_1.setAutoDraw(False)
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in image3Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "image3" ---
    for thisComponent in image3Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('image3.stopped', globalClock.getTime())
    # the Routine "image3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "text6" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('text6.started', globalClock.getTime())
    advanceScreen_10.keys = []
    advanceScreen_10.rt = []
    _advanceScreen_10_allKeys = []
    # keep track of which components have finished
    text6Components = [advanceScreen_10, directionText_5]
    for thisComponent in text6Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "text6" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *advanceScreen_10* updates
        waitOnFlip = False
        
        # if advanceScreen_10 is starting this frame...
        if advanceScreen_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            advanceScreen_10.frameNStart = frameN  # exact frame index
            advanceScreen_10.tStart = t  # local t and not account for scr refresh
            advanceScreen_10.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(advanceScreen_10, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'advanceScreen_10.started')
            # update status
            advanceScreen_10.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(advanceScreen_10.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(advanceScreen_10.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if advanceScreen_10.status == STARTED and not waitOnFlip:
            theseKeys = advanceScreen_10.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
            _advanceScreen_10_allKeys.extend(theseKeys)
            if len(_advanceScreen_10_allKeys):
                advanceScreen_10.keys = _advanceScreen_10_allKeys[-1].name  # just the last key pressed
                advanceScreen_10.rt = _advanceScreen_10_allKeys[-1].rt
                advanceScreen_10.duration = _advanceScreen_10_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *directionText_5* updates
        
        # if directionText_5 is starting this frame...
        if directionText_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            directionText_5.frameNStart = frameN  # exact frame index
            directionText_5.tStart = t  # local t and not account for scr refresh
            directionText_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(directionText_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'directionText_5.started')
            # update status
            directionText_5.status = STARTED
            directionText_5.setAutoDraw(True)
        
        # if directionText_5 is active this frame...
        if directionText_5.status == STARTED:
            # update params
            pass
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in text6Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "text6" ---
    for thisComponent in text6Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('text6.stopped', globalClock.getTime())
    # the Routine "text6" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "image4" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('image4.started', globalClock.getTime())
    advanceScreen_11.keys = []
    advanceScreen_11.rt = []
    _advanceScreen_11_allKeys = []
    # keep track of which components have finished
    image4Components = [advanceScreen_11, cueExampleImage_2]
    for thisComponent in image4Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "image4" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *advanceScreen_11* updates
        waitOnFlip = False
        
        # if advanceScreen_11 is starting this frame...
        if advanceScreen_11.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            advanceScreen_11.frameNStart = frameN  # exact frame index
            advanceScreen_11.tStart = t  # local t and not account for scr refresh
            advanceScreen_11.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(advanceScreen_11, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'advanceScreen_11.started')
            # update status
            advanceScreen_11.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(advanceScreen_11.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(advanceScreen_11.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if advanceScreen_11.status == STARTED and not waitOnFlip:
            theseKeys = advanceScreen_11.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
            _advanceScreen_11_allKeys.extend(theseKeys)
            if len(_advanceScreen_11_allKeys):
                advanceScreen_11.keys = _advanceScreen_11_allKeys[-1].name  # just the last key pressed
                advanceScreen_11.rt = _advanceScreen_11_allKeys[-1].rt
                advanceScreen_11.duration = _advanceScreen_11_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *cueExampleImage_2* updates
        
        # if cueExampleImage_2 is starting this frame...
        if cueExampleImage_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cueExampleImage_2.frameNStart = frameN  # exact frame index
            cueExampleImage_2.tStart = t  # local t and not account for scr refresh
            cueExampleImage_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cueExampleImage_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'cueExampleImage_2.started')
            # update status
            cueExampleImage_2.status = STARTED
            cueExampleImage_2.setAutoDraw(True)
        
        # if cueExampleImage_2 is active this frame...
        if cueExampleImage_2.status == STARTED:
            # update params
            pass
        
        # if cueExampleImage_2 is stopping this frame...
        if cueExampleImage_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > cueExampleImage_2.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                cueExampleImage_2.tStop = t  # not accounting for scr refresh
                cueExampleImage_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cueExampleImage_2.stopped')
                # update status
                cueExampleImage_2.status = FINISHED
                cueExampleImage_2.setAutoDraw(False)
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in image4Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "image4" ---
    for thisComponent in image4Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('image4.stopped', globalClock.getTime())
    # the Routine "image4" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "text7" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('text7.started', globalClock.getTime())
    advanceScreen_12.keys = []
    advanceScreen_12.rt = []
    _advanceScreen_12_allKeys = []
    # keep track of which components have finished
    text7Components = [advanceScreen_12, directionText_6]
    for thisComponent in text7Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "text7" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *advanceScreen_12* updates
        waitOnFlip = False
        
        # if advanceScreen_12 is starting this frame...
        if advanceScreen_12.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            advanceScreen_12.frameNStart = frameN  # exact frame index
            advanceScreen_12.tStart = t  # local t and not account for scr refresh
            advanceScreen_12.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(advanceScreen_12, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'advanceScreen_12.started')
            # update status
            advanceScreen_12.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(advanceScreen_12.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(advanceScreen_12.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if advanceScreen_12.status == STARTED and not waitOnFlip:
            theseKeys = advanceScreen_12.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
            _advanceScreen_12_allKeys.extend(theseKeys)
            if len(_advanceScreen_12_allKeys):
                advanceScreen_12.keys = _advanceScreen_12_allKeys[-1].name  # just the last key pressed
                advanceScreen_12.rt = _advanceScreen_12_allKeys[-1].rt
                advanceScreen_12.duration = _advanceScreen_12_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *directionText_6* updates
        
        # if directionText_6 is starting this frame...
        if directionText_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            directionText_6.frameNStart = frameN  # exact frame index
            directionText_6.tStart = t  # local t and not account for scr refresh
            directionText_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(directionText_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'directionText_6.started')
            # update status
            directionText_6.status = STARTED
            directionText_6.setAutoDraw(True)
        
        # if directionText_6 is active this frame...
        if directionText_6.status == STARTED:
            # update params
            pass
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in text7Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "text7" ---
    for thisComponent in text7Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('text7.stopped', globalClock.getTime())
    # the Routine "text7" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "text8" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('text8.started', globalClock.getTime())
    advanceScreen_13.keys = []
    advanceScreen_13.rt = []
    _advanceScreen_13_allKeys = []
    # keep track of which components have finished
    text8Components = [advanceScreen_13, directionText_7]
    for thisComponent in text8Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "text8" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *advanceScreen_13* updates
        waitOnFlip = False
        
        # if advanceScreen_13 is starting this frame...
        if advanceScreen_13.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            advanceScreen_13.frameNStart = frameN  # exact frame index
            advanceScreen_13.tStart = t  # local t and not account for scr refresh
            advanceScreen_13.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(advanceScreen_13, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'advanceScreen_13.started')
            # update status
            advanceScreen_13.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(advanceScreen_13.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(advanceScreen_13.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if advanceScreen_13.status == STARTED and not waitOnFlip:
            theseKeys = advanceScreen_13.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
            _advanceScreen_13_allKeys.extend(theseKeys)
            if len(_advanceScreen_13_allKeys):
                advanceScreen_13.keys = _advanceScreen_13_allKeys[-1].name  # just the last key pressed
                advanceScreen_13.rt = _advanceScreen_13_allKeys[-1].rt
                advanceScreen_13.duration = _advanceScreen_13_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *directionText_7* updates
        
        # if directionText_7 is starting this frame...
        if directionText_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            directionText_7.frameNStart = frameN  # exact frame index
            directionText_7.tStart = t  # local t and not account for scr refresh
            directionText_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(directionText_7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'directionText_7.started')
            # update status
            directionText_7.status = STARTED
            directionText_7.setAutoDraw(True)
        
        # if directionText_7 is active this frame...
        if directionText_7.status == STARTED:
            # update params
            pass
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in text8Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "text8" ---
    for thisComponent in text8Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('text8.stopped', globalClock.getTime())
    # the Routine "text8" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    runs = data.TrialHandler(nReps=1, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('conditions/PlanetRuns.xlsx'),
        seed=None, name='runs')
    thisExp.addLoop(runs)  # add the loop to the experiment
    thisRun = runs.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisRun.rgb)
    if thisRun != None:
        for paramName in thisRun:
            globals()[paramName] = thisRun[paramName]
    
    for runNumber, thisRun in enumerate(runs, start=1):
        # If relaunching after a crash, skip fully completed earlier rounds.
        # Skipped rounds are not written into this resumed output file.
        if runNumber < START_ROUND:
            continue
        currentLoop = runs
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisRun.rgb)
        if thisRun != None:
            for paramName in thisRun:
                globals()[paramName] = thisRun[paramName]
        
        # --- Prepare to start Routine "trigger" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trigger.started', globalClock.getTime())
        receiveTrigger.keys = []
        receiveTrigger.rt = []
        _receiveTrigger_allKeys = []
        # keep track of which components have finished
        triggerComponents = [waitForTrigger, receiveTrigger]
        for thisComponent in triggerComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trigger" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *waitForTrigger* updates
            
            # if waitForTrigger is starting this frame...
            if waitForTrigger.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                waitForTrigger.frameNStart = frameN  # exact frame index
                waitForTrigger.tStart = t  # local t and not account for scr refresh
                waitForTrigger.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(waitForTrigger, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'waitForTrigger.started')
                # update status
                waitForTrigger.status = STARTED
                waitForTrigger.setAutoDraw(True)
            
            # if waitForTrigger is active this frame...
            if waitForTrigger.status == STARTED:
                # update params
                pass
            
            # *receiveTrigger* updates
            waitOnFlip = False
            
            # if receiveTrigger is starting this frame...
            if receiveTrigger.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                receiveTrigger.frameNStart = frameN  # exact frame index
                receiveTrigger.tStart = t  # local t and not account for scr refresh
                receiveTrigger.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(receiveTrigger, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'receiveTrigger.started')
                # update status
                receiveTrigger.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(receiveTrigger.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(receiveTrigger.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if receiveTrigger.status == STARTED and not waitOnFlip:
                theseKeys = receiveTrigger.getKeys(keyList=['equal'], ignoreKeys=None, waitRelease=False)
                _receiveTrigger_allKeys.extend(theseKeys)
                if len(_receiveTrigger_allKeys):
                    receiveTrigger.keys = _receiveTrigger_allKeys[-1].name  # just the last key pressed
                    receiveTrigger.rt = _receiveTrigger_allKeys[-1].rt
                    receiveTrigger.duration = _receiveTrigger_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in triggerComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trigger" ---
        for thisComponent in triggerComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('trigger.stopped', globalClock.getTime())
        # the Routine "trigger" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "waitForScanner" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('waitForScanner.started', globalClock.getTime())
        # Run 'Begin Routine' code from wait_for_scanner_code
        #if expInfo['mriMode'] != 'Off':
        #    if trigger == 'usb':
        #        vol = launchScan(win, MR_settings, 
        #              globalClock=fmriClock, 
        #              mode=expInfo['mriMode'])
        #        if ser: # for GSR sync
        #            ser.writeUSB(0,1)  # Raise BNC
        #            ser.writeUSB(1,int('11111111',2))  # Raise all pins for Parallel
        
        #    elif trigger == 'parallel':
        #        address = 0x378
        #        #parallel.setPortAddress(0x378)
        #        wait_msg = "Get Ready!"
        #        pinStatus = winioport.inp(address)
        #        waitMsgStim = visual.TextStim(win, color='white', text=wait_msg)
        #        waitMsgStim.draw()
        #        win.flip()
        #        while True:
        #            if pinStatus != winioport.inp(address):
        #               break
                       # start exp when pin values change
        #        fmriClock.reset()
        #        logging.exp('parallel trigger: start of scan')
        #        win.flip()  # blank the screen on first sync pulse received
        
        #expInfo['triggerWallTime'] = time.ctime()
        
        # Run 'Begin Routine' code from logVersionName
        runs.addData('TaskVersion', runOrderFile)
        # keep track of which components have finished
        waitForScannerComponents = []
        for thisComponent in waitForScannerComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "waitForScanner" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in waitForScannerComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "waitForScanner" ---
        for thisComponent in waitForScannerComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('waitForScanner.stopped', globalClock.getTime())
        # Run 'End Routine' code from wait_for_scanner_code
        routineTimer.reset()
        # the Routine "waitForScanner" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        blocks = data.TrialHandler(nReps=1, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions(runOrderFile),
            seed=None, name='blocks')
        thisExp.addLoop(blocks)  # add the loop to the experiment
        thisBlock = blocks.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
        if thisBlock != None:
            for paramName in thisBlock:
                globals()[paramName] = thisBlock[paramName]
        
        for thisBlock in blocks:
            currentLoop = blocks
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
            if thisBlock != None:
                for paramName in thisBlock:
                    globals()[paramName] = thisBlock[paramName]
            
            # --- Prepare to start Routine "trial_cue" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('trial_cue.started', globalClock.getTime())
            # Run 'Begin Routine' code from code
            if Run=='Run1':
                valueConditions=ValueCondition1
            if Run=='Run2':
                valueConditions=ValueCondition2
            if Run=='Run3':
                valueConditions=ValueCondition3
            if Run=='Run4':
                valueConditions=ValueCondition4
            
            
            
            # Set Cue image on Reward Value Condition
            if (valueConditions == 'Low1') or (valueConditions == 'Low2') or (valueConditions == 'Low3') or (valueConditions == 'Low4') or (valueConditions == 'Low5') or (valueConditions=='Low6'):
                valueCondition='low'
                cueImage = 'low.png'
            else:
                valueCondition='high'
                cueImage = 'high.png'
            # set border width of 2nd border frame
            if valueCondition == 'low':
                 borderColor='black'
            else:
                 borderColor='white'
            frame2.setLineColor(borderColor)
            
            
            
            
            # Run 'Begin Routine' code from timingCode
            #cueTimeAdded = False
            currentLoop.addData('blockOnset', fmriClock.getTime())
            currentLoop.addData('cueOnset', fmriClock.getTime())
            Cue2.setImage(os.path.join('images',cueImage))
            # Run 'Begin Routine' code from generateJitterList
            ISIs=copy.copy(allISIs)
            random.shuffle(ISIs)
            # Run 'Begin Routine' code from setStimRotations
            AllRotations=copy.copy(rotations)
            random.shuffle(AllRotations)
            # Run 'Begin Routine' code from blockTiming
            blockClock.reset()
            # keep track of which components have finished
            trial_cueComponents = [Cue2, frame1, frame2, drawCircle]
            for thisComponent in trial_cueComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial_cue" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from timingCode
                #if not cueTimeAdded:
                #    currentLoop.addData('blockOnset', fmriClock.getTime())
                #    currentLoop.addData('cueOnset', fmriClock.getTime())
                #    cueTimeAdded = True
                
                # *Cue2* updates
                
                # if Cue2 is starting this frame...
                if Cue2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    Cue2.frameNStart = frameN  # exact frame index
                    Cue2.tStart = t  # local t and not account for scr refresh
                    Cue2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Cue2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Cue2.started')
                    # update status
                    Cue2.status = STARTED
                    Cue2.setAutoDraw(True)
                
                # if Cue2 is active this frame...
                if Cue2.status == STARTED:
                    # update params
                    pass
                
                # if Cue2 is stopping this frame...
                if Cue2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > Cue2.tStartRefresh + 1-frameTolerance:
                        # keep track of stop time/frame for later
                        Cue2.tStop = t  # not accounting for scr refresh
                        Cue2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Cue2.stopped')
                        # update status
                        Cue2.status = FINISHED
                        Cue2.setAutoDraw(False)
                
                # *frame1* updates
                
                # if frame1 is starting this frame...
                if frame1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    frame1.frameNStart = frameN  # exact frame index
                    frame1.tStart = t  # local t and not account for scr refresh
                    frame1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(frame1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'frame1.started')
                    # update status
                    frame1.status = STARTED
                    frame1.setAutoDraw(True)
                
                # if frame1 is active this frame...
                if frame1.status == STARTED:
                    # update params
                    pass
                
                # if frame1 is stopping this frame...
                if frame1.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > frame1.tStartRefresh + 1-frameTolerance:
                        # keep track of stop time/frame for later
                        frame1.tStop = t  # not accounting for scr refresh
                        frame1.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'frame1.stopped')
                        # update status
                        frame1.status = FINISHED
                        frame1.setAutoDraw(False)
                
                # *frame2* updates
                
                # if frame2 is starting this frame...
                if frame2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    frame2.frameNStart = frameN  # exact frame index
                    frame2.tStart = t  # local t and not account for scr refresh
                    frame2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(frame2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'frame2.started')
                    # update status
                    frame2.status = STARTED
                    frame2.setAutoDraw(True)
                
                # if frame2 is active this frame...
                if frame2.status == STARTED:
                    # update params
                    pass
                
                # if frame2 is stopping this frame...
                if frame2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > frame2.tStartRefresh + 1-frameTolerance:
                        # keep track of stop time/frame for later
                        frame2.tStop = t  # not accounting for scr refresh
                        frame2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'frame2.stopped')
                        # update status
                        frame2.status = FINISHED
                        frame2.setAutoDraw(False)
                
                # *drawCircle* updates
                
                # if drawCircle is starting this frame...
                if drawCircle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    drawCircle.frameNStart = frameN  # exact frame index
                    drawCircle.tStart = t  # local t and not account for scr refresh
                    drawCircle.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(drawCircle, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'drawCircle.started')
                    # update status
                    drawCircle.status = STARTED
                    drawCircle.setAutoDraw(True)
                
                # if drawCircle is active this frame...
                if drawCircle.status == STARTED:
                    # update params
                    pass
                
                # if drawCircle is stopping this frame...
                if drawCircle.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > drawCircle.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        drawCircle.tStop = t  # not accounting for scr refresh
                        drawCircle.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'drawCircle.stopped')
                        # update status
                        drawCircle.status = FINISHED
                        drawCircle.setAutoDraw(False)
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial_cueComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial_cue" ---
            for thisComponent in trial_cueComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('trial_cue.stopped', globalClock.getTime())
            # Run 'End Routine' code from code
            blocks.addData('valueConditions', valueConditions)
            blocks.addData('valueCondition', valueCondition)
            blocks.addData('cueImage', cueImage)
            # Run 'End Routine' code from timingCode
            currentLoop.addData('cueOffset', fmriClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            
            # --- Prepare to start Routine "fixation" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('fixation.started', globalClock.getTime())
            # Run 'Begin Routine' code from fixationFrames
            # set border color of 2nd border frame
            if valueCondition == 'low':
                 borderColor='black'
            else:
                 borderColor='white'
            fixationFrame2.setLineColor(borderColor)
            
            
            
            fixation1.setColor('white', colorSpace='rgb')
            fixation1.setText('+')
            # keep track of which components have finished
            fixationComponents = [fixation1, fixationFrame1, fixationFrame2]
            for thisComponent in fixationComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "fixation" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *fixation1* updates
                
                # if fixation1 is starting this frame...
                if fixation1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    fixation1.frameNStart = frameN  # exact frame index
                    fixation1.tStart = t  # local t and not account for scr refresh
                    fixation1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fixation1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation1.started')
                    # update status
                    fixation1.status = STARTED
                    fixation1.setAutoDraw(True)
                
                # if fixation1 is active this frame...
                if fixation1.status == STARTED:
                    # update params
                    pass
                
                # if fixation1 is stopping this frame...
                if fixation1.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > fixation1.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        fixation1.tStop = t  # not accounting for scr refresh
                        fixation1.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'fixation1.stopped')
                        # update status
                        fixation1.status = FINISHED
                        fixation1.setAutoDraw(False)
                
                # *fixationFrame1* updates
                
                # if fixationFrame1 is starting this frame...
                if fixationFrame1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    fixationFrame1.frameNStart = frameN  # exact frame index
                    fixationFrame1.tStart = t  # local t and not account for scr refresh
                    fixationFrame1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fixationFrame1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixationFrame1.started')
                    # update status
                    fixationFrame1.status = STARTED
                    fixationFrame1.setAutoDraw(True)
                
                # if fixationFrame1 is active this frame...
                if fixationFrame1.status == STARTED:
                    # update params
                    pass
                
                # if fixationFrame1 is stopping this frame...
                if fixationFrame1.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > fixationFrame1.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        fixationFrame1.tStop = t  # not accounting for scr refresh
                        fixationFrame1.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'fixationFrame1.stopped')
                        # update status
                        fixationFrame1.status = FINISHED
                        fixationFrame1.setAutoDraw(False)
                
                # *fixationFrame2* updates
                
                # if fixationFrame2 is starting this frame...
                if fixationFrame2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    fixationFrame2.frameNStart = frameN  # exact frame index
                    fixationFrame2.tStart = t  # local t and not account for scr refresh
                    fixationFrame2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fixationFrame2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixationFrame2.started')
                    # update status
                    fixationFrame2.status = STARTED
                    fixationFrame2.setAutoDraw(True)
                
                # if fixationFrame2 is active this frame...
                if fixationFrame2.status == STARTED:
                    # update params
                    pass
                
                # if fixationFrame2 is stopping this frame...
                if fixationFrame2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > fixationFrame2.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        fixationFrame2.tStop = t  # not accounting for scr refresh
                        fixationFrame2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'fixationFrame2.stopped')
                        # update status
                        fixationFrame2.status = FINISHED
                        fixationFrame2.setAutoDraw(False)
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in fixationComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "fixation" ---
            for thisComponent in fixationComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('fixation.stopped', globalClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.000000)
            
            # set up handler to look after randomisation of conditions etc
            trials = data.TrialHandler(nReps=1, method='sequential', 
                extraInfo=expInfo, originPath=-1,
                trialList=data.importConditions(colorTrialsFile),
                seed=None, name='trials')
            thisExp.addLoop(trials)  # add the loop to the experiment
            thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
            if thisTrial != None:
                for paramName in thisTrial:
                    globals()[paramName] = thisTrial[paramName]
            
            for thisTrial in trials:
                currentLoop = trials
                thisExp.timestampOnFlip(win, 'thisRow.t')
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        inputs=inputs, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                )
                # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
                if thisTrial != None:
                    for paramName in thisTrial:
                        globals()[paramName] = thisTrial[paramName]
                
                # --- Prepare to start Routine "trial" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('trial.started', globalClock.getTime())
                # Run 'Begin Routine' code from setJitter
                ISI=ISIs.pop()
                # Run 'Begin Routine' code from setRotation
                rotation=AllRotations.pop()
                # Run 'Begin Routine' code from setTargetColors
                if valueConditions=="Low1":
                    targetColor=LowColors1
                    targetType=TrialType1
                    prepot=Prepot1
                    GoPrepot=GoPrepot1
                    stimPic=LowStimPic1
                elif valueConditions=="High1":
                    targetColor=HighColors1
                    targetType=TrialType1
                    prepot=Prepot1
                    GoPrepot=GoPrepot1
                    stimPic=HighStimPic1
                if valueConditions=="Low2":
                    targetColor=LowColors2
                    targetType=TrialType2
                    prepot=Prepot2
                    GoPrepot=GoPrepot2
                    stimPic=LowStimPic2
                elif valueConditions=="High2":
                    targetColor=HighColors2
                    targetType=TrialType2
                    prepot=Prepot2
                    GoPrepot=GoPrepot2
                    stimPic=HighStimPic2
                if valueConditions=="Low3":
                    targetColor=LowColors3
                    targetType=TrialType3
                    prepot=Prepot3
                    GoPrepot=GoPrepot3
                    stimPic=LowStimPic3
                elif valueConditions=="High3":
                    targetColor=HighColors3
                    targetType=TrialType3
                    prepot=Prepot3
                    GoPrepot=GoPrepot3
                    stimPic=HighStimPic3
                if valueConditions=="Low4":
                    targetColor=LowColors4
                    targetType=TrialType4
                    prepot=Prepot4
                    GoPrepot=GoPrepot4
                    stimPic=LowStimPic4
                elif valueConditions=="High4":
                    targetColor=HighColors4
                    targetType=TrialType4
                    prepot=Prepot4
                    GoPrepot=GoPrepot4
                    stimPic=HighStimPic4
                if valueConditions=="Low5":
                    targetColor=LowColors5
                    targetType=TrialType5
                    prepot=Prepot5
                    GoPrepot=GoPrepot5
                    stimPic=HighStimPic5
                elif valueConditions=="High5":
                    targetColor=HighColors5
                    targetType=TrialType5
                    prepot=Prepot5
                    GoPrepot=GoPrepot5
                    stimPic=HighStimPic5
                if valueConditions=="Low6":
                    targetColor=LowColors6
                    targetType=TrialType6
                    prepot=Prepot6
                    GoPrepot=GoPrepot6
                    stimPic=LowStimPic6
                elif valueConditions=="High6":
                    targetColor=HighColors6
                    targetType=TrialType6
                    prepot=Prepot6
                    GoPrepot=GoPrepot6
                    stimPic=HighStimPic6
                # Run 'Begin Routine' code from setKeyResponses
                #targetImage=stimPic
                
                # Set Target Image on Go/Nogo Condition
                if expInfo['hand'] == 'right':
                    if targetType == 'go':
                      correct =  '1'
                    else:
                      correct = None
                else: # if hand='left'
                    if targetType == 'go':
                      correct =  '6'
                    else:
                      correct = None
                
                # set border color of 2nd border frame
                if valueCondition == 'low':
                     borderColor='black'
                else:
                     borderColor='white'
                trialFrame2.setLineColor(borderColor)
                
                
                Target.setOri(rotation)
                Target.setImage(os.path.join('images',stimPic))
                targetResp.keys = []
                targetResp.rt = []
                _targetResp_allKeys = []
                # Run 'Begin Routine' code from targetTiming
                trials.addData('targetOnset', fmriClock.getTime())
                
                #targetTimeAdded= False
                
                # keep track of which components have finished
                trialComponents = [trialFrame, trialFrame2, Target, targetResp]
                for thisComponent in trialComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "trial" ---
                routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 0.5:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *trialFrame* updates
                    
                    # if trialFrame is starting this frame...
                    if trialFrame.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        trialFrame.frameNStart = frameN  # exact frame index
                        trialFrame.tStart = t  # local t and not account for scr refresh
                        trialFrame.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(trialFrame, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'trialFrame.started')
                        # update status
                        trialFrame.status = STARTED
                        trialFrame.setAutoDraw(True)
                    
                    # if trialFrame is active this frame...
                    if trialFrame.status == STARTED:
                        # update params
                        pass
                    
                    # if trialFrame is stopping this frame...
                    if trialFrame.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > trialFrame.tStartRefresh + 0.5-frameTolerance:
                            # keep track of stop time/frame for later
                            trialFrame.tStop = t  # not accounting for scr refresh
                            trialFrame.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'trialFrame.stopped')
                            # update status
                            trialFrame.status = FINISHED
                            trialFrame.setAutoDraw(False)
                    
                    # *trialFrame2* updates
                    
                    # if trialFrame2 is starting this frame...
                    if trialFrame2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        trialFrame2.frameNStart = frameN  # exact frame index
                        trialFrame2.tStart = t  # local t and not account for scr refresh
                        trialFrame2.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(trialFrame2, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'trialFrame2.started')
                        # update status
                        trialFrame2.status = STARTED
                        trialFrame2.setAutoDraw(True)
                    
                    # if trialFrame2 is active this frame...
                    if trialFrame2.status == STARTED:
                        # update params
                        pass
                    
                    # if trialFrame2 is stopping this frame...
                    if trialFrame2.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > trialFrame2.tStartRefresh + 0.5-frameTolerance:
                            # keep track of stop time/frame for later
                            trialFrame2.tStop = t  # not accounting for scr refresh
                            trialFrame2.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'trialFrame2.stopped')
                            # update status
                            trialFrame2.status = FINISHED
                            trialFrame2.setAutoDraw(False)
                    
                    # *Target* updates
                    
                    # if Target is starting this frame...
                    if Target.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        Target.frameNStart = frameN  # exact frame index
                        Target.tStart = t  # local t and not account for scr refresh
                        Target.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(Target, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Target.started')
                        # update status
                        Target.status = STARTED
                        Target.setAutoDraw(True)
                    
                    # if Target is active this frame...
                    if Target.status == STARTED:
                        # update params
                        pass
                    
                    # if Target is stopping this frame...
                    if Target.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > Target.tStartRefresh + 0.5-frameTolerance:
                            # keep track of stop time/frame for later
                            Target.tStop = t  # not accounting for scr refresh
                            Target.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'Target.stopped')
                            # update status
                            Target.status = FINISHED
                            Target.setAutoDraw(False)
                    
                    # *targetResp* updates
                    waitOnFlip = False
                    
                    # if targetResp is starting this frame...
                    if targetResp.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                        # keep track of start time/frame for later
                        targetResp.frameNStart = frameN  # exact frame index
                        targetResp.tStart = t  # local t and not account for scr refresh
                        targetResp.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(targetResp, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'targetResp.started')
                        # update status
                        targetResp.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(targetResp.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(targetResp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    
                    # if targetResp is stopping this frame...
                    if targetResp.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > targetResp.tStartRefresh + 0.5-frameTolerance:
                            # keep track of stop time/frame for later
                            targetResp.tStop = t  # not accounting for scr refresh
                            targetResp.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'targetResp.stopped')
                            # update status
                            targetResp.status = FINISHED
                            targetResp.status = FINISHED
                    if targetResp.status == STARTED and not waitOnFlip:
                        theseKeys = targetResp.getKeys(keyList=['1', '2', '6', '7'], ignoreKeys=None, waitRelease=False)
                        _targetResp_allKeys.extend(theseKeys)
                        if len(_targetResp_allKeys):
                            targetResp.keys = _targetResp_allKeys[0].name  # just the first key pressed
                            targetResp.rt = _targetResp_allKeys[0].rt
                            targetResp.duration = _targetResp_allKeys[0].duration
                            # was this correct?
                            if (targetResp.keys == str(correct)) or (targetResp.keys == correct):
                                targetResp.corr = 1
                            else:
                                targetResp.corr = 0
                    # Run 'Each Frame' code from targetTiming
                    #if not targetTimeAdded:
                    #    currentLoop.addData('targetOnset', fmriClock.getTime())
                    #    targetTimeAdded = True
                    if defaultKeyboard.getKeys(keyList=["escape"]):
                        thisExp.status = FINISHED
                        endExpNow = True
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, inputs=inputs, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in trialComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "trial" ---
                for thisComponent in trialComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('trial.stopped', globalClock.getTime())
                # Run 'End Routine' code from setJitter
                trials.addData('ISI', ISI)
                # Run 'End Routine' code from setKeyResponses
                trials.addData('valueCondition', valueCondition)
                #trials.addData('targetImage', targetImage)
                trials.addData('targetColor', targetColor)
                trials.addData('stimRotation', rotation)
                trials.addData('thisTargetType', targetType)
                trials.addData('thisTrialNoGoPrepot', prepot)
                trials.addData('thisTrialGoPrepot', GoPrepot)
                
                # check responses
                if targetResp.keys in ['', [], None]:  # No response was made
                    targetResp.keys = None
                    # was no response the correct answer?!
                    if str(correct).lower() == 'none':
                       targetResp.corr = 1;  # correct non-response
                    else:
                       targetResp.corr = 0;  # failed to respond (incorrectly)
                # store data for trials (TrialHandler)
                trials.addData('targetResp.keys',targetResp.keys)
                trials.addData('targetResp.corr', targetResp.corr)
                if targetResp.keys != None:  # we had a response
                    trials.addData('targetResp.rt', targetResp.rt)
                    trials.addData('targetResp.duration', targetResp.duration)
                # Run 'End Routine' code from targetTiming
                trials.addData('targetOffset', fmriClock.getTime())
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if routineForceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-0.500000)
                
                # --- Prepare to start Routine "jitter" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('jitter.started', globalClock.getTime())
                # Run 'Begin Routine' code from jitter_code
                # set border color of 2nd border frame
                if valueCondition == 'low':
                     borderColor='black'
                else:
                     borderColor='white'
                jitterFrame2.setLineColor(borderColor)
                
                # Set Target Image on Go/Nogo Condition
                if expInfo['hand'] == 'right':
                    if targetType == 'go':
                      correct =  '1'
                    else:
                      correct = None
                else: # if hand='left'
                    if targetType == 'go':
                      correct =  '6'
                    else:
                      correct = None
                lateResp.keys = []
                lateResp.rt = []
                _lateResp_allKeys = []
                # keep track of which components have finished
                jitterComponents = [jitteredISI, jitterFrame1, jitterFrame2, lateResp]
                for thisComponent in jitterComponents:
                    thisComponent.tStart = None
                    thisComponent.tStop = None
                    thisComponent.tStartRefresh = None
                    thisComponent.tStopRefresh = None
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED
                # reset timers
                t = 0
                _timeToFirstFrame = win.getFutureFlipTime(clock="now")
                frameN = -1
                
                # --- Run Routine "jitter" ---
                routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *jitteredISI* updates
                    
                    # if jitteredISI is starting this frame...
                    if jitteredISI.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        jitteredISI.frameNStart = frameN  # exact frame index
                        jitteredISI.tStart = t  # local t and not account for scr refresh
                        jitteredISI.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(jitteredISI, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'jitteredISI.started')
                        # update status
                        jitteredISI.status = STARTED
                        jitteredISI.setAutoDraw(True)
                    
                    # if jitteredISI is active this frame...
                    if jitteredISI.status == STARTED:
                        # update params
                        pass
                    
                    # if jitteredISI is stopping this frame...
                    if jitteredISI.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > jitteredISI.tStartRefresh + ISI-frameTolerance:
                            # keep track of stop time/frame for later
                            jitteredISI.tStop = t  # not accounting for scr refresh
                            jitteredISI.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'jitteredISI.stopped')
                            # update status
                            jitteredISI.status = FINISHED
                            jitteredISI.setAutoDraw(False)
                    
                    # *jitterFrame1* updates
                    
                    # if jitterFrame1 is starting this frame...
                    if jitterFrame1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        jitterFrame1.frameNStart = frameN  # exact frame index
                        jitterFrame1.tStart = t  # local t and not account for scr refresh
                        jitterFrame1.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(jitterFrame1, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'jitterFrame1.started')
                        # update status
                        jitterFrame1.status = STARTED
                        jitterFrame1.setAutoDraw(True)
                    
                    # if jitterFrame1 is active this frame...
                    if jitterFrame1.status == STARTED:
                        # update params
                        pass
                    
                    # if jitterFrame1 is stopping this frame...
                    if jitterFrame1.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > jitterFrame1.tStartRefresh + ISI-frameTolerance:
                            # keep track of stop time/frame for later
                            jitterFrame1.tStop = t  # not accounting for scr refresh
                            jitterFrame1.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'jitterFrame1.stopped')
                            # update status
                            jitterFrame1.status = FINISHED
                            jitterFrame1.setAutoDraw(False)
                    
                    # *jitterFrame2* updates
                    
                    # if jitterFrame2 is starting this frame...
                    if jitterFrame2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        jitterFrame2.frameNStart = frameN  # exact frame index
                        jitterFrame2.tStart = t  # local t and not account for scr refresh
                        jitterFrame2.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(jitterFrame2, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'jitterFrame2.started')
                        # update status
                        jitterFrame2.status = STARTED
                        jitterFrame2.setAutoDraw(True)
                    
                    # if jitterFrame2 is active this frame...
                    if jitterFrame2.status == STARTED:
                        # update params
                        pass
                    
                    # if jitterFrame2 is stopping this frame...
                    if jitterFrame2.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > jitterFrame2.tStartRefresh + ISI-frameTolerance:
                            # keep track of stop time/frame for later
                            jitterFrame2.tStop = t  # not accounting for scr refresh
                            jitterFrame2.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'jitterFrame2.stopped')
                            # update status
                            jitterFrame2.status = FINISHED
                            jitterFrame2.setAutoDraw(False)
                    
                    # *lateResp* updates
                    waitOnFlip = False
                    
                    # if lateResp is starting this frame...
                    if lateResp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        lateResp.frameNStart = frameN  # exact frame index
                        lateResp.tStart = t  # local t and not account for scr refresh
                        lateResp.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(lateResp, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'lateResp.started')
                        # update status
                        lateResp.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(lateResp.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(lateResp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    
                    # if lateResp is stopping this frame...
                    if lateResp.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > lateResp.tStartRefresh + 1-frameTolerance:
                            # keep track of stop time/frame for later
                            lateResp.tStop = t  # not accounting for scr refresh
                            lateResp.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'lateResp.stopped')
                            # update status
                            lateResp.status = FINISHED
                            lateResp.status = FINISHED
                    if lateResp.status == STARTED and not waitOnFlip:
                        theseKeys = lateResp.getKeys(keyList=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'space'], ignoreKeys=None, waitRelease=False)
                        _lateResp_allKeys.extend(theseKeys)
                        if len(_lateResp_allKeys):
                            lateResp.keys = _lateResp_allKeys[0].name  # just the first key pressed
                            lateResp.rt = _lateResp_allKeys[0].rt
                            lateResp.duration = _lateResp_allKeys[0].duration
                            # was this correct?
                            if (lateResp.keys == str(correct)) or (lateResp.keys == correct):
                                lateResp.corr = 1
                            else:
                                lateResp.corr = 0
                    # Run 'Each Frame' code from writeFeedbackOutcome
                    # write outcome text
                    if targetResp.keys: # if a response is made during planet presentation
                      if targetResp.corr==1: 
                            if valueCondition == 'low':
                                outcomeText = '+ 200'
                                outcomeNumber=200
                            else:  # valueCondition == 'high'
                                outcomeText = '+ $1000'
                                outcomeNumber=1000
                      else: # if targetResp.corr=0
                        if targetType=='nogo':
                            if valueCondition == 'low':
                                outcomeText = '- 100'
                                outcomeNumber= -100
                            else:  # valueCondition == 'high'
                                outcomeText = '- 500'
                                outcomeNumber= -500
                    
                    if not targetResp.keys: # if no response is made during planet presentation
                      if  targetType=='go':
                        if lateResp.corr==1: # if correct late response
                            if valueCondition == 'low':
                                outcomeText = '+ 200'
                                outcomeNumber=200
                            else:  # valueCondition == 'high'
                                outcomeText = '+ 1000'
                                outcomeNumber=1000 
                        else: # if no response is made
                            if valueCondition == 'low':
                                outcomeText = '- 100'
                                outcomeNumber= -100
                            else:  # valueCondition == 'high'
                                outcomeText = '- 500'
                                outcomeNumber= -500
                    
                      elif targetType == 'nogo': # if no response is made for a nogo trial
                        rt_array = np.array(lateResp.rt)
                        if rt_array.size == 0:
                            if valueCondition == 'low':
                                outcomeText = '+ 200'
                                outcomeNumber=200
                            else: # if high
                                outcomeText = '+ 1000'
                                outcomeNumber=1000
                        else: # if a late response is made for a nogo
                            if valueCondition == 'low':
                                outcomeText = '- 100'
                                outcomeNumber= -100
                            else:  # valueCondition == 'high'
                                outcomeText = '- 500'
                                outcomeNumber= -500
                    
                    
                    # label accuracy including late responses
                    if outcomeNumber==1000 or outcomeNumber==200:
                        Accuracy=1
                    else:
                        Accuracy=0
                    
                    # Run 'Each Frame' code from recordRT
                    if targetResp.keys:
                        RT=targetResp.rt
                    else:
                        RT=np.add(lateResp.rt, 0.5)
                    #    RT=0.5 + (lateResp.rt)
                    
                    if targetType=='go':
                        goRT=RT
                    else:
                        goRT='nan'
                    
                    
                    
                    if thisExp.status == FINISHED or endExpNow:
                        endExperiment(thisExp, inputs=inputs, win=win)
                        return
                    
                    # check if all components have finished
                    if not continueRoutine:  # a component has requested a forced-end of Routine
                        routineForceEnded = True
                        break
                    continueRoutine = False  # will revert to True if at least one component still running
                    for thisComponent in jitterComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "jitter" ---
                for thisComponent in jitterComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('jitter.stopped', globalClock.getTime())
                # check responses
                if lateResp.keys in ['', [], None]:  # No response was made
                    lateResp.keys = None
                    # was no response the correct answer?!
                    if str(correct).lower() == 'none':
                       lateResp.corr = 1;  # correct non-response
                    else:
                       lateResp.corr = 0;  # failed to respond (incorrectly)
                # store data for trials (TrialHandler)
                trials.addData('lateResp.keys',lateResp.keys)
                trials.addData('lateResp.corr', lateResp.corr)
                if lateResp.keys != None:  # we had a response
                    trials.addData('lateResp.rt', lateResp.rt)
                    trials.addData('lateResp.duration', lateResp.duration)
                # Run 'End Routine' code from writeFeedbackOutcome
                trials.addData('Accuracy', Accuracy)
                trials.addData('outcomeText', outcomeText)
                trials.addData('outcomeNumber', outcomeNumber)
                # Run 'End Routine' code from recordRT
                trials.addData('RT', RT)
                trials.addData('goRT', goRT)
                # the Routine "jitter" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                thisExp.nextEntry()
                
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
            # completed 1 repeats of 'trials'
            
            
            # --- Prepare to start Routine "feedback" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('feedback.started', globalClock.getTime())
            # Run 'Begin Routine' code from rewardFrames
            # set border color of 2nd border frame
            if valueCondition == 'low':
                 borderColor='black'
            else:
                 borderColor='white'
            rewardFrame2.setLineColor(borderColor)
            
            # calculate reward total
            outcomeTotal = trials.data['outcomeNumber'].sum()
            if outcomeTotal >= 0:
               if outcomeTotal == 0:
                outcomeType='got'
               else:
                outcomeType='won'
            if outcomeTotal<0:
               outcomeType='lost'
            
            if outcomeType=='won':
                rewardColor='lawngreen'
            else:
                rewardColor='orangered'
            
            outcomeDisplay=abs(outcomeTotal)
            msg = "You %s %s points!" %(outcomeType, outcomeDisplay)
            
            reward.setColor(rewardColor, colorSpace='rgb')
            reward.setText(msg)
            # Run 'Begin Routine' code from feedbackTiming
            blocks.addData('feedbackOnset', fmriClock.getTime())
            # keep track of which components have finished
            feedbackComponents = [reward, rewardFrame1, rewardFrame2]
            for thisComponent in feedbackComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "feedback" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.5:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *reward* updates
                
                # if reward is starting this frame...
                if reward.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    reward.frameNStart = frameN  # exact frame index
                    reward.tStart = t  # local t and not account for scr refresh
                    reward.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(reward, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'reward.started')
                    # update status
                    reward.status = STARTED
                    reward.setAutoDraw(True)
                
                # if reward is active this frame...
                if reward.status == STARTED:
                    # update params
                    pass
                
                # if reward is stopping this frame...
                if reward.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > reward.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        reward.tStop = t  # not accounting for scr refresh
                        reward.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'reward.stopped')
                        # update status
                        reward.status = FINISHED
                        reward.setAutoDraw(False)
                
                # *rewardFrame1* updates
                
                # if rewardFrame1 is starting this frame...
                if rewardFrame1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    rewardFrame1.frameNStart = frameN  # exact frame index
                    rewardFrame1.tStart = t  # local t and not account for scr refresh
                    rewardFrame1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(rewardFrame1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rewardFrame1.started')
                    # update status
                    rewardFrame1.status = STARTED
                    rewardFrame1.setAutoDraw(True)
                
                # if rewardFrame1 is active this frame...
                if rewardFrame1.status == STARTED:
                    # update params
                    pass
                
                # if rewardFrame1 is stopping this frame...
                if rewardFrame1.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > rewardFrame1.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        rewardFrame1.tStop = t  # not accounting for scr refresh
                        rewardFrame1.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'rewardFrame1.stopped')
                        # update status
                        rewardFrame1.status = FINISHED
                        rewardFrame1.setAutoDraw(False)
                
                # *rewardFrame2* updates
                
                # if rewardFrame2 is starting this frame...
                if rewardFrame2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    rewardFrame2.frameNStart = frameN  # exact frame index
                    rewardFrame2.tStart = t  # local t and not account for scr refresh
                    rewardFrame2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(rewardFrame2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rewardFrame2.started')
                    # update status
                    rewardFrame2.status = STARTED
                    rewardFrame2.setAutoDraw(True)
                
                # if rewardFrame2 is active this frame...
                if rewardFrame2.status == STARTED:
                    # update params
                    pass
                
                # if rewardFrame2 is stopping this frame...
                if rewardFrame2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > rewardFrame2.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        rewardFrame2.tStop = t  # not accounting for scr refresh
                        rewardFrame2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'rewardFrame2.stopped')
                        # update status
                        rewardFrame2.status = FINISHED
                        rewardFrame2.setAutoDraw(False)
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in feedbackComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "feedback" ---
            for thisComponent in feedbackComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('feedback.stopped', globalClock.getTime())
            # Run 'End Routine' code from rewardFrames
            blocks.addData('outcomeTotal', outcomeTotal)
            # Run 'End Routine' code from feedbackTiming
            blocks.addData('feedbackOffset', fmriClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.500000)
            
            # --- Prepare to start Routine "endBlock" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('endBlock.started', globalClock.getTime())
            # keep track of which components have finished
            endBlockComponents = [breakFixation]
            for thisComponent in endBlockComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "endBlock" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 10.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *breakFixation* updates
                
                # if breakFixation is starting this frame...
                if breakFixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    breakFixation.frameNStart = frameN  # exact frame index
                    breakFixation.tStart = t  # local t and not account for scr refresh
                    breakFixation.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(breakFixation, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'breakFixation.started')
                    # update status
                    breakFixation.status = STARTED
                    breakFixation.setAutoDraw(True)
                
                # if breakFixation is active this frame...
                if breakFixation.status == STARTED:
                    # update params
                    pass
                
                # if breakFixation is stopping this frame...
                if breakFixation.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > breakFixation.tStartRefresh + 10-frameTolerance:
                        # keep track of stop time/frame for later
                        breakFixation.tStop = t  # not accounting for scr refresh
                        breakFixation.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'breakFixation.stopped')
                        # update status
                        breakFixation.status = FINISHED
                        breakFixation.setAutoDraw(False)
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in endBlockComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "endBlock" ---
            for thisComponent in endBlockComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('endBlock.stopped', globalClock.getTime())
            # Run 'End Routine' code from blockOffset
            blocks.addData('blockOffset', fmriClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-10.000000)
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1 repeats of 'blocks'
        
        
        # --- Prepare to start Routine "runEnd" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('runEnd.started', globalClock.getTime())
        SpaceToEndRun.keys = []
        SpaceToEndRun.rt = []
        _SpaceToEndRun_allKeys = []
        # Run 'Begin Routine' code from runOffset
        runs.addData('runOffset', fmriClock.getTime())
        # keep track of which components have finished
        runEndComponents = [breakText, SpaceToEndRun]
        for thisComponent in runEndComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "runEnd" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *breakText* updates
            
            # if breakText is starting this frame...
            if breakText.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                breakText.frameNStart = frameN  # exact frame index
                breakText.tStart = t  # local t and not account for scr refresh
                breakText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(breakText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'breakText.started')
                # update status
                breakText.status = STARTED
                breakText.setAutoDraw(True)
            
            # if breakText is active this frame...
            if breakText.status == STARTED:
                # update params
                pass
            
            # *SpaceToEndRun* updates
            waitOnFlip = False
            
            # if SpaceToEndRun is starting this frame...
            if SpaceToEndRun.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                SpaceToEndRun.frameNStart = frameN  # exact frame index
                SpaceToEndRun.tStart = t  # local t and not account for scr refresh
                SpaceToEndRun.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(SpaceToEndRun, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'SpaceToEndRun.started')
                # update status
                SpaceToEndRun.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(SpaceToEndRun.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(SpaceToEndRun.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if SpaceToEndRun.status == STARTED and not waitOnFlip:
                theseKeys = SpaceToEndRun.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
                _SpaceToEndRun_allKeys.extend(theseKeys)
                if len(_SpaceToEndRun_allKeys):
                    SpaceToEndRun.keys = _SpaceToEndRun_allKeys[-1].name  # just the last key pressed
                    SpaceToEndRun.rt = _SpaceToEndRun_allKeys[-1].rt
                    SpaceToEndRun.duration = _SpaceToEndRun_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in runEndComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "runEnd" ---
        for thisComponent in runEndComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('runEnd.stopped', globalClock.getTime())
        # the Routine "runEnd" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        # Save the normal PsychoPy output after each completed round.
        # This creates/updates the current segment file only; it does not append to prior files.
        saveData(thisExp=thisExp)
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1 repeats of 'runs'
    
    
    # --- Prepare to start Routine "End" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('End.started', globalClock.getTime())
    SpaceToExit.keys = []
    SpaceToExit.rt = []
    _SpaceToExit_allKeys = []
    # keep track of which components have finished
    EndComponents = [endTask, SpaceToExit]
    for thisComponent in EndComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "End" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *endTask* updates
        
        # if endTask is starting this frame...
        if endTask.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            endTask.frameNStart = frameN  # exact frame index
            endTask.tStart = t  # local t and not account for scr refresh
            endTask.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(endTask, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'endTask.started')
            # update status
            endTask.status = STARTED
            endTask.setAutoDraw(True)
        
        # if endTask is active this frame...
        if endTask.status == STARTED:
            # update params
            pass
        
        # *SpaceToExit* updates
        waitOnFlip = False
        
        # if SpaceToExit is starting this frame...
        if SpaceToExit.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            SpaceToExit.frameNStart = frameN  # exact frame index
            SpaceToExit.tStart = t  # local t and not account for scr refresh
            SpaceToExit.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(SpaceToExit, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'SpaceToExit.started')
            # update status
            SpaceToExit.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(SpaceToExit.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(SpaceToExit.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if SpaceToExit.status == STARTED and not waitOnFlip:
            theseKeys = SpaceToExit.getKeys(keyList=['space'], ignoreKeys=None, waitRelease=False)
            _SpaceToExit_allKeys.extend(theseKeys)
            if len(_SpaceToExit_allKeys):
                SpaceToExit.keys = _SpaceToExit_allKeys[-1].name  # just the last key pressed
                SpaceToExit.rt = _SpaceToExit_allKeys[-1].rt
                SpaceToExit.duration = _SpaceToExit_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in EndComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "End" ---
    for thisComponent in EndComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('End.stopped', globalClock.getTime())
    # the Routine "End" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    # Run 'End Experiment' code from wait_for_scanner_code
    #ser.writeUSB(0,0)
    #ser.writeUSB(1,0)
    #ser.close()
    #win.close()
    #core.quit()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment.

    This version is intentionally written to keep ONE normal file set per launch.
    It can be called after each completed round for crash safety, but it saves to
    the same current-launch CSV and psydat names each time instead of creating a
    separate file per round.
    """
    filename = thisExp.dataFileName
    csv_path = filename + '.csv'
    psydat_path = filename + '.psydat'

    # Write to temporary names first, then replace the current-launch files.
    # This avoids PsychoPy file-collision renaming during round-level autosaves.
    tmp_csv = filename + '_autosave_tmp.csv'
    tmp_pickle_stem = filename + '_autosave_tmp'
    tmp_psydat = tmp_pickle_stem + '.psydat'

    # Remove stale temp files from a prior interrupted save, if present.
    for path in [tmp_csv, tmp_psydat]:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    # Save CSV to temp, then replace/update the one current-launch CSV.
    try:
        thisExp.saveAsWideText(tmp_csv, delim='auto')
        if os.path.exists(tmp_csv):
            os.replace(tmp_csv, csv_path)
    except TypeError:
        # Some older PsychoPy versions have slightly different signatures.
        thisExp.saveAsWideText(tmp_csv)
        if os.path.exists(tmp_csv):
            os.replace(tmp_csv, csv_path)

    # Save psydat to temp, then replace/update the one current-launch psydat.
    try:
        thisExp.saveAsPickle(tmp_pickle_stem)
        if os.path.exists(tmp_psydat):
            os.replace(tmp_psydat, psydat_path)
    except Exception:
        # Do not crash the task just because the backup psydat save failed.
        # The CSV is the main analysis file and has already been saved above.
        logging.warn('Autosave psydat failed; CSV autosave completed.')

def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
