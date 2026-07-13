const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('trafficDesk', {
  getRuntimeInfo: () => ipcRenderer.invoke('app:get-runtime-info'),
  selectVideoFile: () => ipcRenderer.invoke('dialog:select-video-file'),
  restartBackend: () => ipcRenderer.invoke('backend:restart')
});


