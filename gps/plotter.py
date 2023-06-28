"""
Lots taken from: https://stackoverflow.com/questions/7490491/capture-embedded-google-map-image-with-python
"""
from io import BytesIO
import numpy as np
import os
from PIL import Image
import requests
import sys


EARTH_RADIUS = 6378137
EQUATOR_CIRCUMFERENCE = 2 * np.pi * EARTH_RADIUS
INITIAL_RESOLUTION = EQUATOR_CIRCUMFERENCE / 256.0
ORIGIN_SHIFT = EQUATOR_CIRCUMFERENCE / 2.0

from gps.conversions import latlong_to_utm, utm_to_latlong


class GPSPlotter(object):

    def __init__(self,
                 nw_latlong=(37.915585, -122.336621),
                 se_latlong=(37.914514, -122.334064),
                 zoom=19,
                 satellite_img_fname=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rfs_satellite.png'),
                 google_maps_api_key="AIzaSyCj77sQFM1rcIjh0G2ksNFMxzph9jWnmSE"):
        self.nw_latlong = nw_latlong
        self.se_latlong = se_latlong
        self._zoom = zoom
        self._satellite_img_fname = satellite_img_fname
        self._google_maps_api_key = google_maps_api_key
        
        # if not os.path.exists(self._satellite_img_fname):
        #     assert google_maps_api_key is not None
        #     self._save_satellite_image(google_maps_api_key)
        assert google_maps_api_key is not None
        self._update_satellite_image(google_maps_api_key, save_fig=True)
        assert os.path.exists(self._satellite_img_fname)
        self._img = np.array(Image.open(self._satellite_img_fname))

        x_c0, y_c0 = self.latlong_to_pixels(*self.nw_latlong)
        x_c1, y_c1 = self.latlong_to_pixels(*self.se_latlong)

        self._bottom_left_pixel = np.array([min(x_c0, x_c1), min(y_c0, y_c1)])
        self._top_right_pixel = np.array([max(x_c0, x_c1), max(y_c0, y_c1)])

        self._plt_latlong_and_compass_bearing_dict = dict()
        self._plt_latlong_dict = dict()
        self._plt_utms_dicts = dict()

    @property
    def satellite_image(self):
        return self._img.copy()

    def latlong_to_coordinate(self, latlong):
        latlong = np.array(latlong)
        if len(latlong.shape) > 1:
            return np.array([self.latlong_to_coordinate(l_i) for l_i in latlong])

        pixel_absolute = np.array(self.latlong_to_pixels(*latlong))
        # assert np.all(pixel_absolute >= self._bottom_left_pixel) and np.all(pixel_absolute <= self._top_right_pixel)
        pixel = pixel_absolute - self._bottom_left_pixel
        return pixel

    def utm_to_coordinate(self, utm):
        return self.latlong_to_coordinate(utm_to_latlong(utm))

    def compass_bearing_to_dcoord(self, compass_bearing):
        offset = -np.pi / 2.
        dx, dy = np.array([np.cos(compass_bearing + offset), -np.sin(compass_bearing + offset)])
        return np.array([dx, dy])

    def plot_latlong_and_compass_bearing(self, ax, latlong, compass_bearing, blit=True, color='r'):
        return self.plot_utm_and_compass_bearing(ax, latlong_to_utm(latlong), compass_bearing, blit=blit, color=color)

    def plot_latlong(self, ax, latlong, blit=True, colors=['r'], labels=[''],
                     point_size=20, font_size=10, remove_other_latlong=False,
                     adaptive_satellite_img=True):
        if adaptive_satellite_img:
            max_latlong = latlong.max(axis=0)
            min_latlong = latlong.min(axis=0)

            self.nw_latlong = (max_latlong[0] + 1e-5, min_latlong[1] - 1e-5)
            self.se_latlong = (min_latlong[0] - 1e-5, max_latlong[1] + 1e-5)

            assert self._google_maps_api_key is not None
            self._img = self._update_satellite_image(self._google_maps_api_key)

            x_c0, y_c0 = self.latlong_to_pixels(*self.nw_latlong)
            x_c1, y_c1 = self.latlong_to_pixels(*self.se_latlong)

            self._bottom_left_pixel = np.array([min(x_c0, x_c1), min(y_c0, y_c1)])
            self._top_right_pixel = np.array([max(x_c0, x_c1), max(y_c0, y_c1)])

        x, y = np.split(self.utm_to_coordinate(latlong_to_utm(latlong)), 2, axis=-1)

        if ax not in self._plt_latlong_dict:
            imshow = ax.imshow(np.flipud(self._img), origin='lower')
            # points = ax.scatter(x, y, color=colors, label=label, s=point_size)
            points = []
            texts = []
            for idx, (x_, y_) in enumerate(zip(x, y)):
                point = ax.scatter(x_, y_, color=colors[idx], label=labels[idx], s=point_size)
                points.append(point)

                text = ax.text(x_ + 0.20, y_ - 0.20, str(idx), fontsize=font_size)
                texts.append(text)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            self._plt_latlong_dict[ax] = (imshow, points, texts)
            # self._plt_latlong_dict[ax] = (imshow, points)
        else:
            imshow, points, texts = self._plt_latlong_dict[ax]
            if remove_other_latlong:
                points.remove()
                for text in texts:
                    text.remove()
            # imshow, points = self._plt_latlong_dict[ax]
            # if remove_other_latlong:
            #     points.remove()
            # points = ax.scatter(x, y, color=color, label=label, s=point_size)
            # texts = []
            # for idx, (x_, y_) in enumerate(zip(x, y)):
            #     text = ax.text(x_, y_, str(idx))
            #     texts.append(text)
            points = []
            texts = []
            for idx, (x_, y_) in enumerate(zip(x, y)):
                point = ax.scatter(x_, y_, color=colors[idx], label=labels[idx], s=point_size)
                points.append(point)

                text = ax.text(x_ + 0.20, y_ - 0.20, str(idx), fontsize=font_size)
                texts.append(text)
            self._plt_latlong_dict[ax] = (imshow, points, texts)
            # self._plt_latlong_dict[ax] = (imshow, points)

            if blit:
                ax.draw_artist(ax.patch)
                ax.draw_artist(imshow)
                for point, text in zip(points, texts):
                    ax.draw_artist(point)
                    ax.draw_artist(text)
                ax.figure.canvas.blit(ax.bbox)

    def plot_utm_and_compass_bearing(self, ax, utm, compass_bearing,
                                     blit=True, color='r', arrow_length=15, arrow_head_width=10):
        x, y = np.split(self.utm_to_coordinate(utm), 2, axis=-1)
        dx, dy = arrow_length * self.compass_bearing_to_dcoord(compass_bearing)

        if ax not in self._plt_latlong_and_compass_bearing_dict:
            imshow = ax.imshow(np.flipud(self._img), origin='lower')
            arrow = ax.quiver(
                x, y, dx, dy, color=color, headwidth=arrow_head_width)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            self._plt_latlong_and_compass_bearing_dict[ax] = (imshow, arrow)
        else:
            imshow, arrow = self._plt_latlong_and_compass_bearing_dict[ax]
            arrow.remove()
            arrow = ax.quiver(
                x, y, dx, dy, color=color, headwidth=arrow_head_width)
            self._plt_latlong_and_compass_bearing_dict[ax] = (imshow, arrow)

            if blit:
                ax.draw_artist(ax.patch)
                ax.draw_artist(imshow)
                ax.draw_artist(arrow)
                ax.figure.canvas.blit(ax.bbox)

    def plot_latlong_density(self, ax, latlongs, include_image=True, include_colorbar=False, **kwargs):
        xy = filter(lambda x: x is not None,
                    [self.latlong_to_coordinate(latlong) for latlong in latlongs])
        xy = np.array(list(xy))

        if include_image:
            ax.imshow(np.flipud(np.array(Image.fromarray(self._img).convert('L'))), cmap='gray', origin='lower')

        gridsize = 20
        hb = ax.hexbin(
            xy[:, 0], xy[:, 1],
            gridsize=gridsize,
            **kwargs
        )

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if include_colorbar:
            cb = ax.figure.colorbar(hb, ax=ax)

    #######################
    ### Google maps API ###
    #######################

    def latlong_to_pixels(self, lat, lon):
        mx = (lon * ORIGIN_SHIFT) / 180.0
        my = np.log(np.tan((90 + lat) * np.pi / 360.0)) / (np.pi / 180.0)
        my = (my * ORIGIN_SHIFT) / 180.0
        res = INITIAL_RESOLUTION / (2 ** self._zoom)
        px = (mx + ORIGIN_SHIFT) / res
        py = (my + ORIGIN_SHIFT) / res
        return px, py

    def pixels_to_latlong(self, px, py):
        res = INITIAL_RESOLUTION / (2 ** self._zoom)
        mx = px * res - ORIGIN_SHIFT
        my = py * res - ORIGIN_SHIFT
        lat = (my / ORIGIN_SHIFT) * 180.0
        lat = 180 / np.pi * (2 * np.arctan(np.exp(lat * np.pi / 180.0)) - np.pi / 2.0)
        lon = (mx / ORIGIN_SHIFT) * 180.0
        return lat, lon

    def _update_satellite_image(self, google_maps_api_key, save_fig=False):

        ullat, ullon = self.nw_latlong
        lrlat, lrlon = self.se_latlong

        # Set some important parameters
        scale = 1
        maxsize = 640

        # convert all these coordinates to pixels
        ulx, uly = self.latlong_to_pixels(ullat, ullon)
        lrx, lry = self.latlong_to_pixels(lrlat, lrlon)

        # calculate total pixel dimensions of final image
        dx, dy = lrx - ulx, uly - lry

        # calculate rows and columns
        cols, rows = int(np.ceil(dx / maxsize)), int(np.ceil(dy / maxsize))

        # calculate pixel dimensions of each small image
        bottom = 120
        largura = int(np.ceil(dx / cols))
        altura = int(np.ceil(dy / rows))
        alturaplus = altura + bottom

        # assemble the image from stitched
        final = Image.new("RGB", (int(dx), int(dy)))
        for x in range(cols):
            for y in range(rows):
                dxn = largura * (0.5 + x)
                dyn = altura * (0.5 + y)
                latn, lonn = self.pixels_to_latlong(ulx + dxn, uly - dyn - bottom / 2)
                position = ','.join((str(latn), str(lonn)))
                print(x, y, position)
                urlparams = {'center': position,
                             'zoom': str(self._zoom),
                             'size': '%dx%d' % (largura, alturaplus),
                             'maptype': 'satellite',
                             'sensor': 'false',
                             'scale': scale}
                if google_maps_api_key is not None:
                    urlparams['key'] = google_maps_api_key

                url = 'http://maps.google.com/maps/api/staticmap'
                try:
                    response = requests.get(url, params=urlparams)
                    response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    print(e)
                    sys.exit(1)

                im = Image.open(BytesIO(response.content))
                final.paste(im, (int(x * largura), int(y * altura)))

        if save_fig:
            final.save(self._satellite_img_fname)

        return final


class AllRfsGPSPlotter(GPSPlotter):

    def __init__(self, google_maps_api_key=None):
        super(AllRfsGPSPlotter, self).__init__(
            nw_latlong=(37.920236, -122.339465),
            se_latlong=(37.911224, -122.329813),
            satellite_img_fname=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'all_rfs_satellite.png'),
            google_maps_api_key=google_maps_api_key
        )
