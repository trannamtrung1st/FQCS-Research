using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace FQCS.OpenCVSharp
{
    public partial class ShowCamForm : Form
    {
        private VideoCapture _vidCapture;
        public ShowCamForm()
        {
            InitializeComponent();
            this.FormClosed += ShowCamForm_FormClosed;
        }

        private void ShowCamForm_FormClosed(object sender, FormClosedEventArgs e)
        {
            if (_vidCapture != null)
                _vidCapture.Dispose();
        }

        protected void ShowCam(string ipAddr)
        {
            if (_vidCapture != null) _vidCapture.Dispose();
            _vidCapture = string.IsNullOrWhiteSpace(ipAddr) ?
                new VideoCapture(0) : new VideoCapture(ipAddr);

            using (Window window = new Window("Camera"))
            using (Mat image = new Mat()) // Frame image buffer
            {
                // When the movie playback reaches end, Mat.data becomes NULL.
                while (Cv2.GetWindowProperty("Camera", WindowProperty.AutoSize) != -1)
                {
                    _vidCapture.Read(image); // same as cvQueryFrame
                    if (image.Empty()) break;
                    window.ShowImage(image);
                    Cv2.WaitKey(30);
                }
            }
        }

        private void btnShow_Click(object sender, EventArgs e)
        {
            string ipAddr = txtIpAddress.Text;
            ShowCam(ipAddr);
        }
    }
}
