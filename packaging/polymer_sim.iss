; Inno Setup script for Polymer Growth Simulator.
;
; Build with:
;   ISCC.exe /DAppVersion=0.1.0 packaging\polymer_sim.iss
;
; Reads from:  dist\Polymer Growth Simulator\
; Outputs to:  dist\PolymerGrowthSimulator-Setup-<version>.exe

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

#define AppName       "Polymer Growth Simulator"
#define AppPublisher  "Kaan Basaran"
#define AppExe        "PolymerGrowthSimulator.exe"
#define AppId         "{{B5E9A2C0-7C9F-4A1E-9B6E-POLYMERGROWTH00}}"

[Setup]
AppId={#AppId}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=..\dist
OutputBaseFilename=PolymerGrowthSimulator-Setup-{#AppVersion}
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
UninstallDisplayIcon={app}\{#AppExe}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
; Source path is relative to this .iss file (packaging/). dist/ is one level up.
Source: "..\dist\Polymer Growth Simulator\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
