Set oWS = WScript.CreateObject("WScript.Shell")
sLinkFile = oWS.SpecialFolders("Desktop") & "\Stellar Logic AI Dashboard.lnk"
Set oLink = oWS.CreateShortcut(sLinkFile)
oLink.TargetPath = "c:\Users\merce\Documents\helm-ai\dashboard_server.py"
oLink.Arguments = ""
oLink.WorkingDirectory = "c:\Users\merce\Documents\helm-ai"
oLink.IconLocation = "c:\Users\merce\Documents\helm-ai\Stellar_Logic_AI_Logo.png"
oLink.Description = "Launch Stellar Logic AI Executive Dashboard & Assistant"
oLink.Save
